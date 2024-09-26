#import ragnews
import json
#import rag, ArticleDB 

class RAGEvaluator:
    def __init__(self, db_path='ragnews.db'):
        '''
        Initialize the RAGEvaluator with a database connection.
        '''
        self.db = ArticleDB(db_path)
 
    def predict(self, masked_text):
        valid_labels = ['Harris', 'Trump', 'Biden']
        textprompt = f"""
        I will give you a sentence with masked tokens, such as [MASK0], [MASK1], etc. Your job is to predict the correct value of each masked token.
        - Only provide valid values: {valid_labels}.
        - Do not provide any explanation or extra words, only the predicted values separated by a space.

        INPUT: {masked_text}
        OUTPUT:
        """

        # Get the model's prediction
        output = ragnews.rag(textprompt, self.db, keywords_text=masked_text)

        # If the output is empty, retry or provide a fallback
        if not output.strip():
            print(f"Empty prediction for: {masked_text}")
            return ['']  # Return a list with an empty string to avoid errors later

        return output.strip()
   

def load_data(filepath):
    '''
    Load the data from the provided file.
    '''
    with open(filepath, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def evaluate_predictions(data, db_path='ragnews.db'):
    evaluator = RAGEvaluator(db_path)
    total = len(data)
    correct = 0

    for entry in data:
        masked_text = entry['masked_text']
        expected_masks = entry['masks']

        # Get the prediction
        prediction = evaluator.predict(masked_text).split()  # Split in case of multiple masks

        # Normalize both predictions and expected masks
        normalized_prediction = [p.lower() for p in prediction]
        normalized_expected = [e.lower() for e in expected_masks]

        # Check for empty predictions
        if not normalized_prediction:  # If the prediction is empty
            print(f"Empty prediction!\nMasked text: {masked_text}\nExpected: {normalized_expected}\nPredicted: {normalized_prediction}\n")
            continue  # Skip to the next entry

        # Check if all expected masks are in the predictions
        if all(mask in normalized_prediction for mask in normalized_expected):
            correct += 1
        else:
            missing_masks = [mask for mask in normalized_expected if mask not in normalized_prediction]
            print(f"NOT CORRECT\nMasked text: {masked_text}\nExpected: {normalized_expected}\nPredicted: {normalized_prediction}\nMissing: {missing_masks}\n")

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Load the data from the file
    filepath = 'hairy-trumpet/data/wiki__page=2024_United_States_elections,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]'
    data = load_data(filepath)

    # Evaluate the predictions
    evaluate_predictions(data, db_path='ragnews.db')
