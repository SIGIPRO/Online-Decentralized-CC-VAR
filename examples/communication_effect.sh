cd ..

# # Data effect
python3 -m examples.error_comparison experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1 protocol.C_data=10 protocol.C_param=100 

python3 -m examples.dynamic_regret experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1 protocol.C_data=10 protocol.C_param=100 

# Parameter effect
python3 -m examples.error_comparison experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1 protocol.C_data=1 protocol.C_param=100

python3 -m examples.dynamic_regret experiment=error_comparison 'experiment.run.enabled_cases=[global,pure_local,parameter_dataset]' model.algorithmParam.Tstep=1 protocol.C_data=1 protocol.C_param=100