import subprocess

def test_query(query, expected_substrings):
    """
    Run ragnews.py with a query and check if the expected substrings are present in the result.
    """
    result = subprocess.run(
        ['python3', 'ragnews.py', '--query', query],
        capture_output=True,
        text=True
    )
    
    # Check if the script ran successfully
    if result.returncode != 0:
        print(f"Script failed with error: {result.stderr}")
        raise AssertionError(f"Script failed with error: {result.stderr}")
    
    print(f"Output for query '{query}':\n{result.stdout}")
    
    # Check if all expected substrings are present in the output
    for expected_substring in expected_substrings:
        assert expected_substring in result.stdout, f"Expected substring '{expected_substring}' not found in output. Got: {result.stdout}"

def main():
    # Define key phrases that should be present in the output
    test_query("Who is the current democratic presidential nominee?", [
        "Democratic presidential nominee",
        "Kamala Harris"
    ])
    # Add more test cases as needed

if __name__ == "__main__":
    main()

