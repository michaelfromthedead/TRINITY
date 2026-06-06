//! Blackbox tests for trinity-contracts with macro usage.

use trinity_contracts::{contract, Contract, ContractSchema, Constraint};

// Test the contract macro on actual functions
#[contract]
fn double(x: i32) -> i32 {
    x * 2
}

#[contract]
fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[contract]
fn no_return() {
    let _ = 1 + 1;
}

#[test]
fn test_contracted_double() {
    assert_eq!(double(5), 10);
    assert_eq!(double(0), 0);
    assert_eq!(double(-3), -6);
}

#[test]
fn test_contracted_add() {
    assert_eq!(add(2, 3), 5);
    assert_eq!(add(-1, 1), 0);
}

#[test]
fn test_contracted_no_return() {
    no_return();
}

#[test]
fn test_contract_workflow() {
    // Create a contract
    let contract = Contract::new("calculate")
        .requires(Constraint::new("input > 0"))
        .requires(Constraint::new("input < 1000"))
        .ensures(Constraint::new("result >= input"));

    assert_eq!(contract.requires.len(), 2);
    assert_eq!(contract.ensures.len(), 1);
}

#[test]
fn test_schema_roundtrip() {
    let mut schema = ContractSchema::new();
    
    schema.add(
        Contract::new("func_a")
            .requires(Constraint::new("x > 0"))
            .ensures(Constraint::new("result > x")),
    );
    
    schema.add(
        Contract::new("func_b")
            .requires(Constraint::new("y != 0")),
    );

    // Serialize
    let json = schema.to_json().expect("Serialization failed");
    
    // Deserialize
    let parsed = ContractSchema::from_json(&json).expect("Deserialization failed");
    
    assert_eq!(parsed.contracts.len(), 2);
    assert_eq!(parsed.contracts[0].function, "func_a");
    assert_eq!(parsed.contracts[1].function, "func_b");
}
