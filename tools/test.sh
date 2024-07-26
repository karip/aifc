
set -e # stop on errors

# ensure clippy gives no errors or warnings
cargo clippy -- -D warnings

cargo doc

# runs all tests

cargo test

# check that some functions never panic
cargo test --release --features internal-no-panic

echo
echo 'Toisto AIFF test suite results:'
echo

cd toisto-aiff-test-suite
python3 toisto-runner.py -c --override-list ../toisto-aifc-override-list.json ../target/debug/examples/aifc-toisto multi

echo

python3 toisto-runner.py -c --override-list ../toisto-aifc-override-list.json ../target/debug/examples/aifc-toisto single

echo
echo "--- All tests OK."
