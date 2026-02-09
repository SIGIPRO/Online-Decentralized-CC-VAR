
cd ..

python3 -m examples.error_comparison experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1

python3 -m examples.dynamic_regret experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1
