import subprocess

def test_query(query):
    # Run the ragnews.py script with the provided query
    result = subprocess.run(['python3', 'ragnews.py', '--query', query],
                            capture_output=True, text=True)

    # Print the actual output for debugging
    print(f"Output for query '{query}':\n{result.stdout}")

    # Ensure the script ran successfully
    assert result.returncode == 0, f"Script failed with error: {result.stderr}"

def main():
    # Test cases where output is not checked, only successful execution
    test_query("Who is the current democratic presidential nominee?")
    # Add more test queries if needed

if __name__ == '__main__':
    main()

