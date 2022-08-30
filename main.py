import pandas as pd
import spacy
import numpy as np

# Task 1: greetings

def do_task_1(df):
    greeting_words = [
        'здравствуй',
        'здравствуйте',
        'добрый день',
        'доброе утро',
        'добрый вечер',
        'приветствую вас',
        'привет',
        'салют',
        'приветик',
        'хэлло',
        'хай',
        'день добрый',
        'утро доброе',
        'вечер добрый',
        'приветствую'
    ]

    # Function to detect a greeting

    def is_greeting(sentence: str, greeting_words: list):
        if any(greeting_word in sentence for greeting_word in greeting_words):
            return True
        return False

    df['is_greeting'] = df['text'].apply(lambda sentence: is_greeting(sentence, greeting_words))

    print('Task 1 done!')
    return df


# Task 2: whether a manager introduced himself or not

def do_task_2(df, nlp):
    # Will use spacy to work with Named Entity Recognizer

    def did_manager_introduce_himself(sentence: str):
        document = nlp(sentence)
        for entity in document.ents:
            if entity.label_ == 'PER':
                return True
        return False

    df['did_manager_introduce_himself'] = df.apply(lambda row: did_manager_introduce_himself(row['text']) if row['role']=='manager' else '',
                                                   axis=1)

    print('Task 2 done!')
    return df


# Task 3: get a manager's name

def do_task_3(df, nlp):
    def get_manager_name(sentence: str):
        document = nlp(sentence)
        for entity in document.ents:
            if entity.label_ == 'PER':
                return entity.text

    df['manager_name'] = df.apply(lambda row: get_manager_name(row['text']) if row['did_manager_introduce_himself']==True else '',
                                  axis=1)

    print('Task 3 done!')
    return df


# Task 4: get a company's name

def do_task_4(df, nlp):
    def get_company_name(sentence: str):
        document = nlp(sentence)
        for entity in document.ents:
            if entity.label_ == 'ORG':
                return entity.text
        return ''

    df['company_name'] = df['text'].apply(lambda sentence: get_company_name(sentence))

    print('Task 4 done!')                                                
    return df


# Task 5: farewells

def do_task_5(df):
    farewell_words = [
        'до свидания',
        'до скорого свидания',
        'до встречи',
        'до завтра',
        'до выходных',
        'всего хорошего',
        'всего доброго',
        'удачи',
        'счастливо',
        'спокойной ночи',
        'доброй ночи',
        'до скорой встречи',
        'до скорого',
        'увидимся завтра',
        'увидимся',
        'всего вам доброго'
    ]

    # Function to detect a farewell

    def is_farewell(sentence: str, farewell_words: list):
        if any(farewell_word in sentence for farewell_word in farewell_words):
            return True
        return False

    df['is_farewell'] = df['text'].apply(lambda sentence: is_farewell(sentence, farewell_words))

    print('Task 5 done!')
    return df


# Task 6: check whether a manager said hello and goodbye

def do_task_6(df):
    def is_polite(greeting: bool, farewell: bool):
        if greeting & farewell:
            return True
        return False

    managers_df = df.groupby(['dlg_id', 'role'], 
                             as_index=False).max()

    managers_df = managers_df[managers_df['role']=='manager'][['dlg_id', 
                                                               'role', 
                                                               'is_greeting', 
                                                               'is_farewell']]

    managers_df['is_polite'] = managers_df.apply(lambda row: is_polite(row['is_greeting'],
                                                                       row['is_farewell']),
                                                 axis=1)

    print('Task 6 done!')
    return managers_df


def main():
    nlp = spacy.load('ru_core_news_sm')

    df = pd.read_csv('test_data.csv')
    df['text'] = df['text'].apply(lambda x: x.lower())

    df = do_task_1(df)
    df = do_task_2(df, nlp)
    df = do_task_3(df, nlp)
    df = do_task_4(df, nlp)
    df = do_task_5(df)
    managers_df = do_task_6(df)

    print(df)
    print(managers_df)

    df.to_csv('final_data_1.csv')
    managers_df.to_csv('final_data_2.csv')


if __name__=='__main__':
    main()





