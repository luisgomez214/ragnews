#!/bin/python3

'''
Run an interactive QA session with news articles using the Groq LLM API and retrieval augmented generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

from urllib.parse import urlparse
import datetime
import logging
import re
import sqlite3

import groq

from groq import Groq
import os


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def run_llm(system, user, model='llama3-8b-8192', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                "role": "user",
                "content": user,
            }
        ],
        model=model,
        seed=seed,
    )
    return chat_completion.choices[0].message.content


def summarize_text(text, seed=None):
    system = 'Summarize the input text below.  Limit the summary to 1 paragraph.  Use an advanced reading level similar to the input text, and ensure that all people, places, and other proper and dates nouns are included in the summary.  The summary should be in English.'
    return run_llm(system, text, seed=seed)


def translate_text(text):
    system = 'You are a professional translator working for the United Nations.  The following document is an important news article that needs to be translated into English.  Provide a professional translation.'
    return run_llm(system, text)


def extract_keywords(text, seed=None):
    r'''
    This is a helper function for RAG.
    Given an input text,
    this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Who is the current democratic presidential nominee?', seed=0)
    'Joe candidate nominee presidential Democrat election primary TBD voting politics'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'Trump Mexican immigrants policy position illegal border control deportation walls'

    Note that the examples above are passing in a seed value for deterministic results.
    In production, you probably do not want to specify the seed.
    '''

    # FIXME:
    # Implement this function.
    # It's okay if you don't get the exact same keywords as me.
    # You probably certainly won't because you probably won't come up with the exact same prompt as me.
    # To make the test cases above pass,
    # you'll have to modify them to be what the output of your prompt provides.
    system = '''
    Extract the most important keywords from the input text. Focus on key topics, people, entities, and policies without any unnecessary text.

    1. Only return the keywords as a single string, separated by spaces.
    2. Do not include any extra text like "Here are the keywords".
    3. Only extract the key terms relevant to the query, without stopwords.

    Example:

    Input: 'Who is the current democratic presidential nominee?'
    Output: 'Joe candidate nominee presidential Democrat election primary TBD voting politics'

    Input: 'What is the policy position of Trump related to illegal Mexican immigrants?'
    Output: 'Trump Mexican immigrants policy position illegal border control deportation walls'

    Now, extract the keywords from this input:
    I repeat only return keywords nothing in addition to it
    '''

    return run_llm(system, text, seed=seed)
################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\W+')
    sql_dewhite = rex.sub(' ', sql)
    logging.debug(f'SQL: {sql_dewhite}')


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    We use this decorator on the add_urls method below to ensure that a webcrawl continues even if there are errors.
    '''
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(str(e))
    return inner_function


################################################################################
# rag
################################################################################

def rag(text, db):
    """
    This function uses Retrieval Augmented Generation (RAG) to create an LLM response for the input text.
    The `db` argument should be an instance of the `ArticleDB` class containing relevant documents.

    Steps:
    1. Extract keywords from the input text.
    2. Retrieve related articles based on those keywords.
    3. Construct a new prompt using the query and the article summaries.
    4. Pass the prompt to the LLM and return the response.
    """

    # Step 1: Extract keywords from the input text
    keywords = extract_keywords(text)

    # Step 2: Use the extracted keywords to find relevant articles in the database
    related_articles = db.find_articles(keywords)

    # Step 3: Compile summaries from the relevant articles
    summaries = "\n\n".join(article['en_summary'] for article in related_articles)

    # Create the LLM prompt by incorporating the user query and the relevant article summaries
    prompt = (
        f"You are skilled at responding concisely with information based on article summaries. "
        f"Below is the user's query followed by summaries of relevant articles. "
        f"Please answer based only on the information in the summaries. Do not speculate.\n\n"
        f"User Query: \"{text}\"\n\n"
        f"Relevant Article Summaries:\n{summaries}"
    )

    # Step 4: Pass the constructed prompt to the LLM and return the generated response
    response = run_llm(system=prompt, user=text)

    return response




class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    The following example shows how to add urls to the database.

    >>> db = ArticleDB()
    >>> len(db)
    0
    >>> db.add_url(ArticleDB._TESTURLS[0])
    >>> len(db)
    1

    Once articles have been added,
    we can search through those articles to find articles about only certain topics.

    >>> articles = db.find_articles('Economía')

    The output is a list of articles that match the search query.
    Each article is represented by a dictionary with a number of fields about the article.

    >>> articles[0]['title']
    'La creación de empleo defrauda en Estados Unidos en agosto y aviva el temor a una recesión | Economía | EL PAÍS'
    >>> articles[0].keys()
    ['rowid', 'rank', 'title', 'publish_date', 'hostname', 'url', 'staleness', 'timebias', 'en_summary', 'text']
    '''

    _TESTURLS = [
        'https://elpais.com/economia/2024-09-06/la-creacion-de-empleo-defrauda-en-estados-unidos-en-agosto-y-aviva-el-fantasma-de-la-recesion.html',
        'https://www.cnn.com/2024/09/06/politics/american-push-israel-hamas-deal-analysis/index.html',
        ]

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory=sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.

        The test below demonstrates that creating a schema on a database that already has the schema will not generate errors.

        >>> db = ArticleDB()
        >>> db._create_schema()
        >>> db._create_schema()
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE articles
            USING FTS5 (
                title,
                text,
                hostname,
                url,
                publish_date,
                crawl_date,
                lang,
                en_translation,
                en_summary
                );
            '''
            self.db.execute(sql)
            self.db.commit()

        # if the database already exists,
        # then do nothing
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed')



    def find_articles(self, search_query, max_results=10, time_bias_alpha=1):
        '''
        Find articles in the database that match the given search query.

        Parameters:
        - search_query: The query to search for in the articles.
        - max_results: The maximum number of results to return (default is 10).
        - time_bias_alpha: A parameter for time bias in ranking (default is 1).

        Returns:
        - A list of articles matching the search query.
        '''

        # Format the search query for full-text search
        if isinstance(search_query, str):
            formatted_query = ' OR '.join(f'"{word}"' for word in search_query.split())
        else:
            formatted_query = search_query

        formatted_query = formatted_query.replace("'", "''")
        logging.debug(f'Search Query: {formatted_query}')

        sql_query = '''
            SELECT rowid, rank, title, publish_date, hostname, url, en_summary, text
            FROM articles
            WHERE articles MATCH ?
            ORDER BY rank DESC
            LIMIT ?
        '''

        # Execute the query and fetch results
        result_set = self.db.execute(sql_query, (formatted_query, max_results)).fetchall()

        # Convert result rows to dictionaries
        articles_list = [dict(row) for row in result_set]

        return articles_list

    @_catch_errors
    def add_url(self, url, recursive_depth=0, allow_dupes=False):
        '''
        Download the url, extract various metainformation, and add the metainformation into the db.

        By default, the same url cannot be added into the database multiple times.

        >>> db = ArticleDB()
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> db.add_url(ArticleDB._TESTURLS[0])
        >>> len(db)
        1

        >>> db = ArticleDB()
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> db.add_url(ArticleDB._TESTURLS[0], allow_dupes=True)
        >>> len(db)
        3

        '''
        from bs4 import BeautifulSoup
        import requests
        import metahtml
        logging.info(f'add_url {url}')

        if not allow_dupes:
            logging.debug(f'checking for url in database')
            sql = '''
            SELECT count(*) FROM articles WHERE url=?;
            '''
            _logsql(sql)
            cursor = self.db.cursor()
            cursor.execute(sql, [url])
            row = cursor.fetchone()
            is_dupe = row[0] > 0
            if is_dupe:
                logging.debug(f'duplicate detected, skipping!')
                return

        logging.debug(f'downloading url')
        try:
            response = requests.get(url)
        except requests.exceptions.MissingSchema:
            # if no schema was provided in the url, add a default
            url = 'https://' + url
            response = requests.get(url)
        parsed_uri = urlparse(url)
        hostname = parsed_uri.netloc

        logging.debug(f'extracting information')
        parsed = metahtml.parse(response.text, url)
        info = metahtml.simplify_meta(parsed)

        if info['type'] != 'article' or len(info['content']['text']) < 100:
            logging.debug(f'not an article... skipping')
            en_translation = None
            en_summary = None
            info['title'] = None
            info['content'] = {'text': None}
            info['timestamp.published'] = {'lo': None}
            info['language'] = None
        else:
            logging.debug('summarizing')
            if not info['language'].startswith('en'):
                en_translation = translate_text(info['content']['text'])
            else:
                en_translation = None
            en_summary = summarize_text(info['content']['text'])

        logging.debug('inserting into database')
        sql = '''
        INSERT INTO articles(title, text, hostname, url, publish_date, crawl_date, lang, en_translation, en_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql, [
            info['title'],
            info['content']['text'], 
            hostname,
            url,
            info['timestamp.published']['lo'],
            datetime.datetime.now().isoformat(),
            info['language'],
            en_translation,
            en_summary,
            ])
        self.db.commit()

        logging.debug('recursively adding more links')
        if recursive_depth > 0:
            for link in info['links.all']:
                url2 = link['href']
                parsed_uri2 = urlparse(url2)
                hostname2 = parsed_uri2.netloc
                if hostname in hostname2 or hostname2 in hostname:
                    self.add_url(url2, recursive_depth-1)
        
    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM articles
        WHERE text IS NOT NULL;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Interactive QA with news articles.')
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='ragnews.db')
    parser.add_argument('--recursive_depth', default=0, type=int)
    parser.add_argument('--add_url', help='Add a URL to the database')
    parser.add_argument('--test_find_articles', action='store_true', help='Test finding articles')
    parser.add_argument('--query', help='Query for interactive QA')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel.upper(),
    )

    db = ArticleDB(args.db)

    if args.add_url:
        db.add_url(args.add_url, recursive_depth=args.recursive_depth, allow_dupes=True)
    elif args.test_find_articles:
        db.add_url(ArticleDB._TESTURLS[0])
        db.add_url(ArticleDB._TESTURLS[1])
        results = db.find_articles("Economía")
        for result in results:
            print(f"Title: {result['title']}")
    elif args.query:
        output = rag(args.query, db)
        print(output)
    else:
        import readline
        while True:
            text = input('ragnews> ')
            if len(text.strip()) > 0:
                output = rag(text, db)
                print(output)




