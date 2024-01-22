import re
import pandas as pd
import numpy as np
import os
from pathlib import Path

def remove_emojis(data):
    """removes emojis from text data.
    """
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
        "]+", re.UNICODE)
    data = re.sub(emoj, ' ', data) # replace emoji with whitespace
    return data


def preprocess_workshop_data(message_col):
    """Creates processed pandas Series (column) from message column.
    """    
    processed = message_col.apply(lambda x: re.sub(r'http\S+|t\.me/\S+', ' ', x))
    # remove emojis
    processed = processed.apply(lambda x: remove_emojis(x))
    # exclude search term
    processed = processed.apply(lambda x: re.sub(r'Lütz\S+', ' ', x))
    # replace !?!
    processed = processed.apply(lambda x: re.sub(r'[.?!]+', '.', x))
    # replace multiple whitespace
    processed = processed.apply(lambda x: re.sub(r'\s{2,}', ' ', x))
    processed = processed.apply(str.strip)
    return processed



def deduplicate_message_lines(df, save_path=False):
    """
    """
    unique_message_parts = df.copy()
    unique_message_parts["message_lines"] = unique_message_parts["message"].str.split("\n")
    # Explode the message lines (this basically creates a new row for each line in message_lines and copy pastes the other columns)
    unique_message_parts_exploded = unique_message_parts.explode("message_lines")
    print(f'all message parts: {len(unique_message_parts_exploded)}')

    # Remove duplicates while ensuring that at least one row for each unique combination of "post_id", "created_at", "username", "message", and "processed" is kept
    unique_message_parts_exploded_dedup = unique_message_parts_exploded.drop_duplicates(subset='message_lines', keep='first')
    print(f'deduplicated message parts: {len(unique_message_parts_exploded_dedup)}')

    #This line of code groups the rows in a Pandas DataFrame by the columns "post_id", "created_at", "username", "message", and "processed". It then applies the `agg()` method to the "message_lines" column, joining the values in each group with a newline character.
    #The resulting DataFrame contains one row for each unique combination of values in the specified columns, with the "message_lines" column containing all the lines of text from the original rows that belong to that group.
    unique_message_parts = unique_message_parts_exploded_dedup.groupby(["post_id", "created_at", "username", "message", "post_id_all"], as_index=False, dropna=False).agg({'message_lines': '\n'.join})
    print(f'deduplicated message lines put together: {len(unique_message_parts)}')

    # Remove rows where the "message_lines" column is empty
    unique_message_parts = unique_message_parts[unique_message_parts["message_lines"] != ""]
    print(f'removed empty messages: {len(unique_message_parts)}')
    #print(unique_message_parts_extended_dedup.head(10))

    unique_message_parts.rename(columns={'message': 'message_old', 'message_lines': 'message'}, inplace=True)
    
    # Save the preprocessed data to a CSV file
    if save_path:
        unique_message_parts.to_csv(save_path)
        
    return unique_message_parts


def topics_for_all_messages(messages_all_df, messages_dedup_df):
    """reworks the topics to all post ids, also duplicated ones
    """
    # make post_id_all lists, if not already happened
    messages_dedup_df['post_id_all'] = messages_dedup_df['post_id_all'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
    messages_all_df = messages_all_df.copy()
    for _, row in messages_dedup_df.iterrows():
        for post_id in row['post_id_all']:
                messages_all_df.loc[int(post_id),'topics'] = row['topics']
    return messages_all_df[['post_id','created_at', 'message','username',
        'views', 'forwards', 'reply_count','topics']]


def create_topic_inspection(topic_model, triangulated_data_final, export_path=False):
         #Create an empty dataframe
    topic_inspection = []
    #Extract the most relevant terms per topic
    for t in range(-1, len(topic_model.get_topic_info()) - 1): # or start from -1 for hdbscan
        topic_terms = topic_model.get_topic(t)
        topic_freqency = topic_model.get_topic_freq(t) #this gets the frequency for the reduced dataset which was basis for the model
        actual_topic_freqency = len(triangulated_data_final[triangulated_data_final["topics"] == t]) #this gets the frequency for the original dataset
        for term in topic_terms:
            topic_inspection.append((t, term[0], topic_freqency, actual_topic_freqency))
    #Create a dataframe from the list
    topic_inspection = pd.DataFrame(topic_inspection)
    #Rename the columns
    topic_inspection = topic_inspection.rename(columns={0: "topic_nr", 1: "term", 2: "frequency_by_model", 3: "frequency_actual"})
    #Join the terms per topic
    topic_inspection = topic_inspection.groupby('topic_nr').agg({'term': ' '.join, 'frequency_by_model': 'first', 'frequency_actual': 'first'}).reset_index()
    ## Add 13 new columns to topic_inspection
    # Create a new DataFrame with 23 empty columns to store representative documents and examples
    empty_df = pd.DataFrame(columns=['Column {}'.format(i) for i in range(1, 24)])
    # Concatenate the empty DataFrame with the original DataFrame
    topic_inspection  = pd.concat([topic_inspection , empty_df], axis=1)

    # Set the random seed to make the sample reproducible
    np.random.seed(42)

    # For each topic, retrieve the 3 most representative documents and additionally sample 10 documents and add them to the DataFrame
    for i in topic_inspection["topic_nr"]:
        subset = triangulated_data_final[triangulated_data_final["topics"] == i]
        sample_subset = subset["message"].sample(20,replace=True)
        topic_inspection.loc[i+1, 'Column 4':'Column 23'] = sample_subset.values
        representative_docs = topic_model.get_representative_docs(i)
        topic_inspection.loc[i+1, 'Column 1':'Column 3'] = representative_docs

    # Use a loop to rename the columns with a pattern
    for i in range(1, 4):
        old_name = 'Column ' + str(i)
        new_name = 'representative doc ' + str(i)
        topic_inspection = topic_inspection.rename(columns={old_name: new_name})

    # Use a loop to rename the columns with a pattern
    for i in range(4, 24):
        old_name = 'Column ' + str(i)
        new_name = 'example ' + str(i -3)
        topic_inspection = topic_inspection.rename(columns={old_name: new_name})

    if export_path:
        topic_inspection.to_csv(export_path /'topic_inspection.csv', encoding='utf-8-sig')



def export_topic_results(topic_model, topics, probs, all_messages,dedup_messages, export_name):
    export_path = Path(f'data/{export_name}')
    export_path.mkdir(exist_ok=True,parents=True)
    dedup_messages['topics'] = topics
    triangulated_data_final = topics_for_all_messages(all_messages, dedup_messages)
    triangulated_data_final.to_csv(export_path / 'all_messages.csv', index=False)
    np.save(export_path / "topics.npy", topics)
    np.save(export_path / "probs.npy", probs)
    create_topic_inspection(topic_model, triangulated_data_final, export_path)

