"""Use the existing balanced-outage evaluator with the ten-shift roster."""
import importlib.util

from campaign import COUNT, TEMPLATE, bind


spec = importlib.util.spec_from_file_location('fleet_template_runtime', TEMPLATE/'fleet_runtime.py')
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)

run = bind(template.run, _validate_suite=bind(template._validate_suite, EVALUATION_SLOTS=COUNT))
