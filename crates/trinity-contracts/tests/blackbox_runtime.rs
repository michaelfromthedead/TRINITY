//! Blackbox tests for runtime contract checking with real use cases.

use trinity_contracts::{contract, ContractChecker};

// Test: Using ContractChecker in real code
mod real_world {
    use super::*;

    fn validate_user_input(age: i32, name: &str) -> Result<(), String> {
        let mut checker = ContractChecker::new("validate_user_input");
        checker
            .requires(age >= 0, "age >= 0")
            .requires(age <= 150, "age <= 150")
            .requires(!name.is_empty(), "name not empty")
            .requires(name.len() <= 100, "name length <= 100");

        if checker.has_violations() {
            let msgs: Vec<_> = checker
                .violations()
                .iter()
                .filter_map(|v| v.message.as_ref())
                .map(|s| s.as_str())
                .collect();
            Err(msgs.join(", "))
        } else {
            Ok(())
        }
    }

    #[test]
    fn test_valid_input() {
        assert!(validate_user_input(25, "Alice").is_ok());
    }

    #[test]
    fn test_invalid_age() {
        assert!(validate_user_input(-5, "Bob").is_err());
        assert!(validate_user_input(200, "Carol").is_err());
    }

    #[test]
    fn test_invalid_name() {
        assert!(validate_user_input(30, "").is_err());
    }
}

// Test: Contract macro with runtime checks
mod macro_integration {
    use super::*;

    #[contract]
    #[requires(x > 0)]
    fn positive_only(x: i32) -> i32 {
        x * 2
    }

    #[contract]
    #[requires(divisor != 0)]
    fn safe_div(dividend: i32, divisor: i32) -> i32 {
        dividend / divisor
    }

    #[test]
    fn test_positive_only() {
        assert_eq!(positive_only(5), 10);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10, 2), 5);
        assert_eq!(safe_div(-10, 5), -2);
    }
}

// Test: State machine with invariants
mod state_machine {
    use super::*;

    struct Counter {
        value: i32,
        max: i32,
    }

    impl Counter {
        fn new(max: i32) -> Self {
            Self { value: 0, max }
        }

        fn increment(&mut self) -> Result<(), &'static str> {
            let mut checker = ContractChecker::new("Counter::increment");
            checker
                .requires(self.value < self.max, "value < max")
                .invariant(self.value >= 0, "value >= 0");

            if checker.has_violations() {
                return Err("Cannot increment");
            }

            self.value += 1;
            Ok(())
        }

        fn decrement(&mut self) -> Result<(), &'static str> {
            let mut checker = ContractChecker::new("Counter::decrement");
            checker.requires(self.value > 0, "value > 0");

            if checker.has_violations() {
                return Err("Cannot decrement");
            }

            self.value -= 1;
            Ok(())
        }
    }

    #[test]
    fn test_counter_valid_ops() {
        let mut counter = Counter::new(5);
        assert!(counter.increment().is_ok());
        assert!(counter.increment().is_ok());
        assert!(counter.decrement().is_ok());
    }

    #[test]
    fn test_counter_bounds() {
        let mut counter = Counter::new(2);
        assert!(counter.increment().is_ok());
        assert!(counter.increment().is_ok());
        assert!(counter.increment().is_err()); // At max

        let mut counter2 = Counter::new(5);
        assert!(counter2.decrement().is_err()); // Already at 0
    }
}

// Test: API validation
mod api_validation {
    use super::*;

    #[derive(Debug)]
    struct ApiRequest {
        method: String,
        path: String,
        body_size: usize,
    }

    fn validate_request(req: &ApiRequest) -> Vec<String> {
        let mut checker = ContractChecker::new("validate_request");

        let valid_methods = ["GET", "POST", "PUT", "DELETE"];
        checker
            .requires(
                valid_methods.contains(&req.method.as_str()),
                "valid HTTP method",
            )
            .requires(req.path.starts_with('/'), "path starts with /")
            .requires(req.body_size <= 10_000_000, "body size <= 10MB");

        checker
            .violations()
            .iter()
            .filter_map(|v| v.message.clone())
            .collect()
    }

    #[test]
    fn test_valid_request() {
        let req = ApiRequest {
            method: "GET".to_string(),
            path: "/api/users".to_string(),
            body_size: 0,
        };
        assert!(validate_request(&req).is_empty());
    }

    #[test]
    fn test_invalid_method() {
        let req = ApiRequest {
            method: "PATCH".to_string(),
            path: "/api/users".to_string(),
            body_size: 0,
        };
        let errors = validate_request(&req);
        assert!(!errors.is_empty());
    }
}
