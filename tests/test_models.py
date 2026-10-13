"""Pure-function tests for the model / spec / registry layers.

These tests deliberately avoid touching UVR's modules: they exercise the
catalog the same way a unit test in :mod:`dorico-maestro.test_commands`
does. They fail on any drift in the field schema, any new architecture
the YAML files don't cover, any required field the spec parses but does
not validate.
"""
