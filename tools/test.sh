
set -e # stop on errors

if [ ! -d "toisto-aiff-test-suite/tests" ]; then
    echo "toisto-aiff-test-suite is missing. Read README.md Testing section for instructions."
    exit 1
fi

# ensure clippy gives no errors or warnings
cargo clippy -- -D warnings

cargo doc

# run tests - no capture if running under GitHub Actions
if [ "$GITHUB_ACTIONS" == "true" ]; then
    cargo test -- --nocapture
else
    cargo test
fi

# build the toisto tester for the test suite
cargo build --example aifc-toisto

# run the toisto test suite
echo
echo 'Toisto AIFF test suite results:'
echo
echo 'multi:'
cd toisto-aiff-test-suite

# target/debug for normal testing,
# x86_64-unknown-linux-gnu and powerpc-unknown-linux-gnu for GitHub Actions
if [ -e ../target/x86_64-unknown-linux-gnu/debug/examples/aifc-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-aifc-override-list.json ../target/x86_64-unknown-linux-gnu/debug/examples/aifc-toisto multi
elif [ -e ../target/powerpc-unknown-linux-gnu/debug/examples/aifc-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-aifc-override-list.json ../target/powerpc-unknown-linux-gnu/debug/examples/aifc-toisto multi
else
    python3 toisto-runner.py -c --override-list ../toisto-aifc-override-list.json ../target/debug/examples/aifc-toisto multi
fi

echo
echo 'single:'

if [ -e ../target/x86_64-unknown-linux-gnu/debug/examples/aifc-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-aifc-override-list.json ../target/x86_64-unknown-linux-gnu/debug/examples/aifc-toisto single
elif [ -e ../target/powerpc-unknown-linux-gnu/debug/examples/aifc-toisto ]; then
    python3 toisto-runner.py -c -v --override-list ../toisto-aifc-override-list.json ../target/powerpc-unknown-linux-gnu/debug/examples/aifc-toisto single
else
    python3 toisto-runner.py -c --override-list ../toisto-aifc-override-list.json ../target/debug/examples/aifc-toisto single
fi

echo
echo "--- All tests OK."
