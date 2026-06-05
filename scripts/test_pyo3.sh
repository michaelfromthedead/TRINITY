#!/bin/bash
# Test script for pyo3 bindings with Python 3.14 compatibility
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo test --package renderer-backend --features pyo3 py_example "$@"
