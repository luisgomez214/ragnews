# RAGClassifier ![](https://github.com/luisgomez214/ragnews/workflows/tests/badge.svg)

This project implements a RAGClassifier following the scikit-learn interface, which uses a Retrieval-Augmented Generation (RAG) system to predict labels for political data.
The RAGClassifier is built to handle the Hairy Trumpet dataset for text classification tasks, leveraging the ragnews.rag function to make predictions. 
The script will parse the data file and extract the possible labels. Then initialize the RAGClassifier over these labels and run the predict method for each instance in the dataset.
Finally, it will compute and display the accuracy of the predictions.


Below is an example of the code. 
```
(venv) luis@Luiss-MacBook-Pro-40 ragnews1 %  python3 ragnews/evaluate.py "hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]"
NOT CORRECT
Masked text: On December 19, 2023, the Colorado Supreme Court removed [MASK0] from the state's 2024 Republican primary, citing the Fourteenth Amendment's ban on candidates who engage in insurrections. This decision was later overturned by the US Supreme Court on March 4, 2024.
Expected: ['trump']
Predicted: ['the', 'court', 'removed', '[[trump]]', 'from', 'the', "state's", '2024', 'republican', 'primary,', 'citing', 'the', 'fourteenth', "amendment's", 'ban', 'on', 'candidates', 'who', 'engage', 'in', 'insurrections.']
Missing: ['trump']

NOT CORRECT
Masked text: After a survey by the Associated Press of Democratic delegates on July 22, 2024, [MASK0] became the new presumptive candidate for the Democratic party, a day after declaring her candidacy. She would become the official nominee on August 5 following a virtual roll call of delegates.
Expected: ['harris']
Predicted: ['katrina', 'never', 'declared', 'candidacy.']
Missing: ['harris']

NOT CORRECT
Masked text: In November 2022, former [MASK0] announced his candidacy in the 2024 presidential election. Other candidates who have entered the 2024 Republican Party presidential primaries include former South Carolina governor and former U.S. Ambassador to the U.N. Nikki Haley and current Florida governor Ron DeSantis, who have since suspended their campaigns. The first Republican presidential debate was held on August 23, 2023, and the first primary contest was the 2024 Iowa Republican presidential caucuses, which was held on January 15, 2024. [MASK0] would win the nomination easily; he was formally nominated at the Republican Convention on July 15, his third consecutive presidential nomination.
Expected: ['trump']
Predicted: ['unfortunately', 'no', 'information', 'suitable', 'for', 'this', 'question', 'could', 'be', 'extracted', 'from', 'the', 'given', 'article', 'summaries', 'because', 'there', 'are', 'actually', 'multiple', 'mentions', 'of', '[mask0].', 'thus,', 'i', 'have', 'to', 'predict', 'two', 'values.', 'the', 'value', 'of', '[mask0]', 'in', 'the', 'first', 'sentence', 'appears', 'to', 'be', "'biden'", '(there', 'is', 'only', 'one', 'mention', 'of', 'any', 'of', 'the', 'values', 'in', 'the', 'text).', 'another', 'mention', 'of', 'the', 'same', 'token', 'later', 'in', 'the', 'text', 'is', 'of', "'trump'"]
Missing: ['trump']

NOT CORRECT
Masked text: All 33 seats in Senate Class 1 and one seat in Senate Class 2 will be up for election; at least one additional special election will take place to fill vacancies that arise during the 118th Congress. Democrats control the majority in the closely-divided Senate following the 2022 U.S. Senate elections, but they will have to defend 23 seats in 2024. Three Democratic-held seats up for election are in the heavily Republican-leaning states of Montana, Ohio, and West Virginia, all of which were won comfortably by [MASK0] in both 2016 and 2020. Other potential Republican targets include seats in Arizona, Michigan, Nevada, Pennsylvania, Wisconsin, and Maryland, while Democrats may target Republican-held seats in Florida and Texas.
Expected: ['trump']
Predicted: ['kamala', 'may', 'be', 'inferred', 'as', "'biden's", 'running', "mate'", 'or', 'such', 'other', 'such', 'answers', "don't", 'include', "'harris'.", 'as', "'biden'", 'ran', 'for', 'president', 'in', '2016,', 'kamala', 'may', 'be', 'inferred', 'as', "'trump's", 'running', "mate.'", 'however', "'biden'", 'being', 'the', 'only', 'candidate', 'found', 'among', "'valid", "values'", ':', 'biden', 'biden']
Missing: ['trump']

Accuracy: 76.47%
```

