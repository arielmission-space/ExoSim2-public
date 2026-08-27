# ExoSim 2.0 - New Test Structure

This directory contains the refactored test structure for ExoSim 2.0, designed to be more maintainable, readable, and scalable.

## Structure Overview

```
test_suite/
├── conftest.py                 # Main test configuration and fixtures
├── unit/                       # Unit tests (fast, isolated)
│   ├── core/                   # Core functionality tests
│   ├── models/                 # Model classes tests
│   ├── tasks/                  # Task-specific tests
│   │   ├── load/               # Loading tasks
│   │   ├── detector/           # Detector-related tasks
│   │   ├── optics/             # Optical elements tasks
│   │   ├── instrument/         # Instrument tasks
│   │   ├── signal_processing/  # Signal processing tasks
│   │   └── tools/              # Tool tasks
│   ├── utils/                  # Utility function tests
│   ├── plots/                  # Plotting functionality tests
│   └── output/                 # Output handling tests
├── integration/                # Integration tests (moderate speed)
│   ├── recipes/                # Recipe integration tests
│   ├── pipelines/              # Pipeline workflow tests
│   └── workflows/              # Complex workflow tests
├── e2e/                        # End-to-end tests (slow, full system)
│   ├── cli/                    # CLI command tests
│   └── scenarios/              # Complete simulation scenarios
├── fixtures/                   # Test fixtures and factories
├── data/                       # Test data files
└── regression/                 # Regression test data and configs
```

## Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Test individual components in isolation
- **Speed**: Fast (< 1s per test)
- **Scope**: Single function/method/class
- **Dependencies**: Minimal, use mocks where needed
- **Examples**: Task parameter validation, signal operations, utility functions

### Integration Tests (`integration/`)
- **Purpose**: Test interaction between components
- **Speed**: Moderate (1-10s per test)
- **Scope**: Multiple components working together
- **Dependencies**: Real components, limited external resources
- **Examples**: Recipe execution, task chains, pipeline segments

### End-to-End Tests (`e2e/`)
- **Purpose**: Test complete workflows from user perspective
- **Speed**: Slow (10s+ per test)
- **Scope**: Full system functionality
- **Dependencies**: Complete system, real data
- **Examples**: CLI commands, full simulation scenarios

## Naming Conventions

### File Naming
- `test_<module_name>.py` - Tests for a specific module
- `test_<functionality>.py` - Tests for specific functionality
- Use descriptive names that clearly indicate what is being tested

### Test Function Naming
- `test_<what_it_does>` - Clear description of what the test verifies
- `test_<component>_<scenario>_<expected_outcome>` - For complex scenarios
- Examples:
  - `test_task_parameter_validation`
  - `test_signal_addition_with_units`
  - `test_recipe_focal_plane_creation_success`

### Test Class Naming
- `Test<ComponentName>` - For testing a specific component
- `Test<Functionality>` - For testing specific functionality
- Examples: `TestTask`, `TestSignalOperations`, `TestFocalPlaneRecipe`

## Fixture Organization

### Global Fixtures (`conftest.py`)
- Session-scoped fixtures for expensive setup
- Common data paths and configurations
- Test environment setup

### Local Fixtures
- Module-specific fixtures in individual test files
- Component-specific test data
- Specialized mocks and stubs

## Running Tests

### Run All Tests
```bash
pytest test_suite/
```

### Run by Category
```bash
# Unit tests only (fast)
pytest test_suite/unit/

# Integration tests
pytest test_suite/integration/

# End-to-end tests
pytest test_suite/e2e/
```

### Run by Component
```bash
# Task-related tests
pytest test_suite/unit/tasks/

# Recipe tests
pytest test_suite/integration/recipes/
```

### Run with Coverage
```bash
pytest test_suite/ --cov=src/exosim --cov-report=html
```

## Migration Notes

This structure replaces the previous flat test structure in `tests/`. Key improvements:

1. **Logical Organization**: Tests grouped by purpose and scope
2. **Parallel Execution**: Different test types can be run independently
3. **Maintainability**: Clear separation makes it easier to maintain tests
4. **Scalability**: Easy to add new test categories or reorganize existing ones
5. **Documentation**: Clear naming and structure makes tests self-documenting

## Best Practices

### Writing Tests
1. **One Assertion Per Test**: Each test should verify one specific behavior
2. **Descriptive Names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use Fixtures**: Leverage pytest fixtures for reusable test setup
5. **Mock External Dependencies**: Use mocks to isolate components under test

### Test Data
1. **Minimal Data**: Use the smallest possible data sets that verify behavior
2. **Shared Fixtures**: Use fixtures for common test data
3. **Regression Data**: Keep regression test data in dedicated directory
4. **Clean State**: Ensure tests don't depend on each other's state

### Performance
1. **Fast Units**: Unit tests should be very fast
2. **Selective Integration**: Only test necessary integration scenarios
3. **Minimal E2E**: Keep end-to-end tests focused on critical paths
4. **Parallel Execution**: Design tests to run in parallel safely
