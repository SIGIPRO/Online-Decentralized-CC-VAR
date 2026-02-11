import numpy as np
from src.core import BaseModel


def _extract_index_list(raw_idx, signal_key):
    if isinstance(raw_idx, dict):
        if signal_key in raw_idx:
            raw_idx = raw_idx[signal_key]
        elif len(raw_idx) == 1:
            raw_idx = next(iter(raw_idx.values()))
        else:
            raise KeyError(f"Missing signal_key={signal_key} in index dict keys={list(raw_idx.keys())}")
    return np.asarray(raw_idx, dtype=int).reshape(-1)


def _build_laplacians_from_complex(cellularComplex, signal_key):
    if cellularComplex is None:
        raise ValueError("cellularComplex is required to infer L_lower/L_upper when not set in algorithmParam")

    if signal_key == 0:
        if 1 not in cellularComplex:
            raise ValueError("Cannot infer node-size Laplacian without B1 in cellularComplex")
        n = int(np.asarray(cellularComplex[1]).shape[0])
    elif signal_key == 1:
        if 1 in cellularComplex:
            n = int(np.asarray(cellularComplex[1]).shape[1])
        elif 2 in cellularComplex:
            n = int(np.asarray(cellularComplex[2]).shape[0])
        else:
            raise ValueError("Cannot infer edge-size Laplacian without B1 or B2 in cellularComplex")
    elif signal_key == 2:
        if 2 not in cellularComplex:
            raise ValueError("Cannot infer 2-cell-size Laplacian without B2 in cellularComplex")
        n = int(np.asarray(cellularComplex[2]).shape[1])
    else:
        raise ValueError(f"Unsupported signal_key={signal_key}; expected 0, 1, or 2")

    L_lower = np.zeros((n, n), dtype=float)
    L_upper = np.zeros((n, n), dtype=float)

    if signal_key in cellularComplex:
        B_down = np.asarray(cellularComplex[signal_key], dtype=float)
        L_lower = B_down.T @ B_down

    if (signal_key + 1) in cellularComplex:
        B_up = np.asarray(cellularComplex[signal_key + 1], dtype=float)
        L_upper = B_up @ B_up.T

    return L_lower, L_upper


class topoLMS:
    """
    Python port of the MATLAB topoLMS core update & prediction.

    Parameters (algorithmParam dict):
        - L_lower: (N x N) numpy array (graph operator for lower part)
        - L_upper: (N x N) numpy array (graph operator for upper part)
        - M:       int, filter order (history length, lower has M, upper has M+1)
        - T:       int, forecast horizon

    Notes
    -----
    This class is **stateless across calls except for**:
        - filter coefficients h (shape: (2*M+1, 1))
        - upper_part  (N x (M+1))
        - lower_part  (N x  M)

    `updateParameters` performs one LMS-style update using the current target
    sample `incomingData["s_next"]` and features from the internally stored
    previous state (CCVAR-like online update order).

    `predictData` returns a T-step-ahead forecast computed by recursively
    rolling the model forward without mutating the trained state.
    """

    def __init__(self, algorithmParam: dict):
        L_lower = algorithmParam.get("L_lower")
        L_upper = algorithmParam.get("L_upper")
        if L_lower is None or L_upper is None:
            raise ValueError("topoLMS requires 'L_lower' and 'L_upper' in algorithmParam")
        if L_lower.shape != L_upper.shape or L_lower.shape[0] != L_lower.shape[1]:
            raise ValueError("L_lower and L_upper must be square matrices of the same size")

        self.N: int = int(L_lower.shape[0])
        self.L_lower: np.ndarray = np.asarray(L_lower, dtype=float)
        self.L_upper: np.ndarray = np.asarray(L_upper, dtype=float)

        self.M: int = int(algorithmParam.get("M", 1))
        self.T: int = int(algorithmParam.get("T", 1))

        # Filter coefficients h = [h_u; h_d]
        self.h_u = np.zeros((self.M + 1, 1), dtype=float)
        self.h_d = np.zeros((self.M, 1), dtype=float) if self.M > 0 else np.zeros((0, 1), dtype=float)
        self.h = np.vstack([self.h_u, self.h_d])  # (2M+1, 1)

        # State buffers for regressor construction
        self.upper_part = np.zeros((self.N, self.M + 1), dtype=float)
        self.lower_part = np.zeros((self.N, self.M), dtype=float) if self.M > 0 else np.zeros((self.N, 0), dtype=float)

        # Last computed prediction (N,)
        self._yhat = np.zeros(self.N, dtype=float)
        # Raw history buffer (CCVAR-like state storage)
        self._data = np.zeros((self.N, self.M + 1), dtype=float)

    # ------------------------- public API -------------------------
    def updateParameters(self, *, incomingData: dict) -> None:
        """
        Perform one parameter update.

        expected keys in incomingData:
          - 's_next': np.ndarray shape (N,) -> current observed sample
        """
        if "s_next" not in incomingData:
            raise KeyError("topoLMS.updateParameters expects incomingData['s_next']")

        x_n = np.asarray(incomingData["s_next"], dtype=float).reshape(self.N)
        y_next = x_n.reshape(self.N, 1)

        # Build X_n from previous state (before injecting x_n), CCVAR-style
        X_n = self._stack_X(self.upper_part, self.lower_part)

        # mu = 1 / (1e-4 + max(eigs(X'X)))  -> spectral norm squared
        # Using SVD for robustness on small regressor dimension.
        if X_n.size:
            svals = np.linalg.svd(X_n, compute_uv=False)
            sigma_max_sq = float(svals[0] ** 2) if svals.size else 0.0
        else:
            sigma_max_sq = 0.0
        mu = 1.0 / (1e-4 + sigma_max_sq)

        # LMS update using previous state to predict current sample.
        self.h -= mu * self._get_gradient(X_n, y_next)

        # Keep explicit raw history buffer (latest sample at last column).
        if self.M > 0:
            self._data[:, 0:self.M] = self._data[:, 1 : self.M + 1]
        self._data[:, -1] = x_n

        # Roll topo states after parameter update.
        self._roll_state(x_n)

    def _get_gradient(self, X_n, y_next):
        return -X_n.T @ (y_next - X_n @ self.h)

    def predictData(self) -> np.ndarray:
        """Return T-step-ahead forecast (N,) without mutating internal state."""
        X_n = self._stack_X(self.upper_part, self.lower_part)
        yhat = self._estimate_t_step(self.T, X_n, self.h, self.L_lower, self.L_upper, self.M)
        self._yhat = yhat
        return yhat

    def _roll_state(self, x_n: np.ndarray) -> None:
        # upper_part = L_upper * upper_part; shift left; last col = x_n
        self.upper_part = self.L_upper @ self.upper_part
        if self.M > 0:
            self.upper_part[:, 0:self.M] = self.upper_part[:, 1:]
        self.upper_part[:, -1] = x_n

        # lower_part update uses current x_n after shift, then applies L_lower
        if self.M > 0:
            if self.M > 1:
                self.lower_part[:, 0 : (self.M - 1)] = self.lower_part[:, 1:]
            self.lower_part[:, -1] = x_n
            self.lower_part = self.L_lower @ self.lower_part

    # ------------------------- helpers -------------------------
    @staticmethod
    def _stack_X(upper_part: np.ndarray, lower_part: np.ndarray) -> np.ndarray:
        if lower_part.size == 0:
            return upper_part
        return np.concatenate([upper_part, lower_part], axis=1)

    @staticmethod
    def _estimate_t_step(step: int, X_n: np.ndarray, h: np.ndarray,
                         L_lower: np.ndarray, L_upper: np.ndarray, M: int) -> np.ndarray:
        """Pure forecast loop: roll forward `step` times using model output as input.
        Mirrors MATLAB estimateTstepMore but without side effects.
        Returns y_hat at horizon as shape (N,).
        """
        upper = X_n[:, : M + 1].copy()
        lower = X_n[:, M + 1 :].copy() if X_n.shape[1] > M + 1 else np.zeros((X_n.shape[0], 0))

        y_hat = None
        for s in range(1, step + 1):
            y_hat = (X_n @ h).reshape(-1)  # (N,)
            if s == step:
                break
            # roll features using predicted signal as the new sample
            # update upper
            upper = L_upper @ upper
            if M > 0:
                upper[:, 0:M] = upper[:, 1:]
            upper[:, -1] = y_hat
            # update lower
            if M > 0:
                if M > 1:
                    lower[:, 0 : (M - 1)] = lower[:, 1:]
                lower = L_lower @ np.column_stack([lower[:, : max(M - 1, 0)], y_hat if M >= 1 else np.array([])]) if M >= 1 else lower
            # rebuild X_n for next inner step
            X_n = np.concatenate([upper, lower], axis=1) if lower.size else upper
        return y_hat if y_hat is not None else np.zeros(X_n.shape[0])

class topoLMSPartial(topoLMS):
    def __init__(self, algorithmParam: dict):
        signal_key = int(algorithmParam.get("signal_key", 1))
        in_idx_raw = algorithmParam.get("in_idx", None)
        out_idx_raw = algorithmParam.get("out_idx", None)

        if in_idx_raw is None:
            raise ValueError("topoLMSPartial requires 'in_idx' in algorithmParam")
        in_idx = _extract_index_list(in_idx_raw, signal_key)
        if in_idx.size == 0:
            raise ValueError("topoLMSPartial received empty in_idx")

        # Build local operators on in_idx.
        L_lower_full = np.asarray(algorithmParam.get("L_lower"), dtype=float)
        L_upper_full = np.asarray(algorithmParam.get("L_upper"), dtype=float)
        if L_lower_full.ndim != 2 or L_lower_full.shape[0] != L_lower_full.shape[1]:
            raise ValueError("L_lower must be a square matrix")
        if L_upper_full.shape != L_lower_full.shape:
            raise ValueError("L_upper must match L_lower shape")

        if np.max(in_idx) >= L_lower_full.shape[0] or np.min(in_idx) < 0:
            raise IndexError("in_idx contains out-of-range entries for provided Laplacians")

        algo_local = dict(algorithmParam)
        algo_local["L_lower"] = L_lower_full[np.ix_(in_idx, in_idx)]
        algo_local["L_upper"] = L_upper_full[np.ix_(in_idx, in_idx)]

        super().__init__(algorithmParam=algo_local)

        self._signal_key = signal_key
        self._in_idx = in_idx

        if out_idx_raw is None:
            out_idx_local = np.arange(self.N, dtype=int)
        else:
            out_idx = _extract_index_list(out_idx_raw, signal_key)
            if out_idx.size == 0:
                out_idx_local = np.arange(self.N, dtype=int)
            elif np.max(out_idx) < self.N and np.min(out_idx) >= 0:
                # already local positions
                out_idx_local = out_idx
            else:
                # treat out_idx as global indices and map to local in_idx positions
                out_pos = []
                in_lookup = {int(g): i for i, g in enumerate(self._in_idx.tolist())}
                for g in out_idx.tolist():
                    if int(g) in in_lookup:
                        out_pos.append(in_lookup[int(g)])
                if len(out_pos) == 0:
                    out_idx_local = np.arange(self.N, dtype=int)
                else:
                    out_idx_local = np.asarray(out_pos, dtype=int)

        self._out_idx = out_idx_local
        self._Nin = int(self.N)
        self._Nout = int(self._out_idx.size)

    def predictData(self, return_full: bool = False) -> np.ndarray:
        yhat_full = super().predictData()
        if return_full:
            return yhat_full
        return yhat_full[self._out_idx]


class topoLMSModel(BaseModel):
    def __init__(self, algorithmParam: dict, cellularComplex=None):
        algorithm_param_local = dict(algorithmParam)
        self._signal_key = int(algorithm_param_local.get("signal_key", 1))
        if ("L_lower" not in algorithm_param_local) or ("L_upper" not in algorithm_param_local):
            L_lower, L_upper = _build_laplacians_from_complex(
                cellularComplex=cellularComplex,
                signal_key=self._signal_key,
            )
            algorithm_param_local.setdefault("L_lower", L_lower)
            algorithm_param_local.setdefault("L_upper", L_upper)

        algorithm = topoLMS(algorithmParam=algorithm_param_local)
        super().__init__(initial_params=algorithm.h.copy(), algorithm=algorithm)
        self._last_mu = 1.0
        self._pending_sample = None

    def _resolve_current_sample(self, aggregated_data):
        data = aggregated_data.get_data()
        if self._signal_key in data:
            x_n = np.asarray(data[self._signal_key], dtype=float).reshape(self._algorithm.N)
            return x_n

        # Fallback: if only one dimension is present, use it as the model signal.
        if len(data) == 1:
            only_key = next(iter(data.keys()))
            self._signal_key = int(only_key)
            x_n = np.asarray(data[only_key], dtype=float).reshape(self._algorithm.N)
            return x_n

        raise KeyError(
            f"topoLMSModel expected signal_key={self._signal_key} in aggregated data keys {list(data.keys())}"
        )

    def get_gradient(self, aggregated_data, **kwargs):
        del kwargs
        x_n = self._resolve_current_sample(aggregated_data=aggregated_data)
        y_next = x_n.reshape(self._algorithm.N, 1)

        X_n = self._algorithm._stack_X(self._algorithm.upper_part, self._algorithm.lower_part)
        if X_n.size:
            svals = np.linalg.svd(X_n, compute_uv=False)
            sigma_max_sq = float(svals[0] ** 2) if svals.size else 0.0
        else:
            sigma_max_sq = 0.0
        self._last_mu = 1.0 / (1e-4 + sigma_max_sq)

        grad = self._algorithm._get_gradient(X_n, y_next).reshape(-1, 1)
        self._pending_sample = x_n
        return grad.flatten()

    def update_params(self, update_term):
        update_arr = np.asarray(update_term, dtype=float).reshape(-1, 1)
        new_params = self._params + self._last_mu * update_arr
        self.set_params(new_params=new_params)

        if self._pending_sample is not None:
            if self._algorithm.M > 0:
                self._algorithm._data[:, 0:self._algorithm.M] = self._algorithm._data[:, 1 : self._algorithm.M + 1]
            self._algorithm._data[:, -1] = self._pending_sample
            self._algorithm._roll_state(self._pending_sample)
            self._pending_sample = None

    def set_params(self, new_params):
        params = np.asarray(new_params, dtype=float).reshape(-1, 1)
        super().set_params(params)
        self._algorithm.h = params.copy()

    def estimate(self, input_data=None, **kwargs):
        del input_data
        steps = max(1, int(kwargs.get("steps", 1)))
        old_t = self._algorithm.T
        self._algorithm.T = steps
        yhat = self._algorithm.predictData()
        self._algorithm.T = old_t
        return {self._signal_key: yhat}


class topoLMSPartialModel(BaseModel):
    def __init__(self, algorithmParam: dict, cellularComplex=None):
        algorithm_param_local = dict(algorithmParam)
        self._signal_key = int(algorithm_param_local.get("signal_key", 1))
        if ("L_lower" not in algorithm_param_local) or ("L_upper" not in algorithm_param_local):
            L_lower, L_upper = _build_laplacians_from_complex(
                cellularComplex=cellularComplex,
                signal_key=self._signal_key,
            )
            algorithm_param_local.setdefault("L_lower", L_lower)
            algorithm_param_local.setdefault("L_upper", L_upper)

        algorithm = topoLMSPartial(algorithmParam=algorithm_param_local)
        super().__init__(initial_params=algorithm.h.copy(), algorithm=algorithm)
        self._last_mu = 1.0
        self._pending_sample = None

    def _resolve_current_sample(self, aggregated_data):
        data = aggregated_data.get_data()
        if self._signal_key in data:
            x_raw = np.asarray(data[self._signal_key], dtype=float).reshape(-1)
        elif len(data) == 1:
            only_key = next(iter(data.keys()))
            self._signal_key = int(only_key)
            x_raw = np.asarray(data[only_key], dtype=float).reshape(-1)
        else:
            raise KeyError(
                f"topoLMSPartialModel expected signal_key={self._signal_key} in aggregated data keys {list(data.keys())}"
            )

        if x_raw.size == self._algorithm.N:
            return x_raw
        if np.max(self._algorithm._in_idx) < x_raw.size:
            return x_raw[self._algorithm._in_idx]
        raise ValueError(
            f"Input sample length {x_raw.size} incompatible with Nin {self._algorithm.N}"
        )

    def get_gradient(self, aggregated_data, **kwargs):
        del kwargs
        x_n = self._resolve_current_sample(aggregated_data=aggregated_data)
        y_next = x_n[self._algorithm._out_idx].reshape(-1, 1)

        X_n = self._algorithm._stack_X(self._algorithm.upper_part, self._algorithm.lower_part)
        X_out = X_n[self._algorithm._out_idx, :]
        if X_out.size:
            svals = np.linalg.svd(X_out, compute_uv=False)
            sigma_max_sq = float(svals[0] ** 2) if svals.size else 0.0
        else:
            sigma_max_sq = 0.0
        self._last_mu = 1.0 / (1e-4 + sigma_max_sq)

        grad = self._algorithm._get_gradient(X_out, y_next).reshape(-1, 1)
        self._pending_sample = x_n
        return grad.flatten()

    def update_params(self, update_term):
        update_arr = np.asarray(update_term, dtype=float).reshape(-1, 1)
        new_params = self._params + self._last_mu * update_arr
        self.set_params(new_params=new_params)

        if self._pending_sample is not None:
            if self._algorithm.M > 0:
                self._algorithm._data[:, 0:self._algorithm.M] = self._algorithm._data[:, 1 : self._algorithm.M + 1]
            self._algorithm._data[:, -1] = self._pending_sample
            self._algorithm._roll_state(self._pending_sample)
            self._pending_sample = None

    def set_params(self, new_params):
        params = np.asarray(new_params, dtype=float).reshape(-1, 1)
        super().set_params(params)
        self._algorithm.h = params.copy()

    def estimate(self, input_data=None, **kwargs):
        del input_data
        steps = max(1, int(kwargs.get("steps", 1)))
        old_t = self._algorithm.T
        self._algorithm.T = steps
        yhat = self._algorithm.predictData(return_full=False)
        self._algorithm.T = old_t
        return {self._signal_key: yhat}
