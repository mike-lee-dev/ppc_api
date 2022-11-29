import pandas as pd
import nltk
from nltk import everygrams
from nltk.corpus import stopwords
from utils import input_output, dataframe
from models import amazon_campaign_link


def phrase_match_suggestion():
    ##  remove stop words
    nltk.download('stopwords')
    dataset = pd.read_excel('./data/raw/Sponsored Products Search term report.xlsx')
    # The subset we're interested in
    data = pd.DataFrame(dataset[['Customer Search Term', 'Impressions', 'Clicks', 'Spend', '7 Day Total Sales ',
                                 '7 Day Total Orders (#)']])

    # data['Number of word'] = data['Customer Search Term'].str.count(' ') + 1
    ## Drop orders less than 1
    new_data = data.drop(data[data['7 Day Total Orders (#)'] <= 0].index)

    ##creating ngrams
    search_terms = list(new_data['Customer Search Term'])
    grams = []
    for i in range(len(search_terms)):
        ## IF you want to remove stop words uncomment the following line and comment the one after
        # search_terms[i] = [word for word in search_terms[i].split() if word not in set(stopwords.words('english')) ]
        search_terms[i] = [word for word in search_terms[i].split()]
        grams.append(list(everygrams(search_terms[i])))
    ## IF you want to limit sequence length add a variable max_len = the limit
    # grams.append(list(everygrams(search_terms[i]), max_len = 5))

    # Create a matching_dict where the key is the Search Sequence and the value is a list
    # The first element is the number of occurances of the sequence and the rest is their indices
    matching_dict = {}
    for instance in grams:
        for gram in instance:
            # Ignore grams that start or end in stop word
            if gram[0] in set(stopwords.words('english')) or gram[-1] in set(stopwords.words('english')):
                continue
            if gram in matching_dict.keys():
                continue
            gram_joined = ' '.join(gram)
            matching_dict[gram_joined] = [-1]
            for i in range(len(grams)):
                # print(gram)
                if gram in grams[i]:
                    matching_dict[gram_joined].append(i)
                    matching_dict[gram_joined][0] += 1

    ## Drop the sequnces that happened only once, and drop the first element (sequence count)
    matching_dict = {key: val[1:] for key, val in matching_dict.items() if val[0] != 0}

    ## drop shorter sequences that occured in longer ones
    for key, val in matching_dict.items():
        for k in matching_dict.keys():
            if key in k and key != k:
                matching_dict[key] = [val for val in matching_dict[key] if val not in matching_dict[k]]

    ## Lose sequences with empty lists
    matching_dict = {key: val for key, val in matching_dict.items() if val}
    matching_dict

    ## Create the final dictionary where the key is the sequence and the values is
    ## [Impressions, Clicks, Spend, 7 Day Total Sales, 7 Day Total Orders (#)]
    final_matching_dict = {}
    for key, val in matching_dict.items():
        final_matching_dict[key] = [sum(new_data.iloc[val]['Impressions']), sum(new_data.iloc[val]['Clicks']),
                                    sum(new_data.iloc[val]['Spend']), sum(new_data.iloc[val]['7 Day Total Sales ']),
                                    sum(new_data.iloc[val]['7 Day Total Orders (#)']),
                                    list(new_data.iloc[val]['Customer Search Term'])]
        final_matching_dict[key] = [round(elem, 2) if isinstance(elem, int) else elem for elem in
                                    final_matching_dict[key]]
    # print(final_matching_dict)
    Final_results = pd.DataFrame.from_dict(final_matching_dict, orient='index')
    Final_results.columns = ['Impressions', 'Clicks', 'Spend', '7 Day Total Sales', '7 Day Total Orders (#)',
                             'Customer Search Terms']
    # Save to excel file
    # Final_results.to_excel('final_results.xlsx')
    Final_results['Number of word'] = Final_results.index.str.count(' ') + 1
    Final_results.reset_index(inplace=True)
    Final_results.sort_values(by='Number of word', inplace=True)
    Final_results.rename(columns={'index': 'Keyword suggestion'}, inplace=True)
    print(Final_results[Final_results['Number of word'] >= 3].sort_values(by='7 Day Total Sales', ascending=False))
    df_bid_SP = input_output.read_bid('SP', path_bid)
    print(df_bid_SP)
    df_bid_SP = dataframe.select_row_by_val(df_bid_SP, 'Match Type', 'phrase')
    df_merge = Final_results[Final_results['Number of word'] >= 3].sort_values(by='7 Day Total Sales', ascending=False)
    df_merge = pd.merge(df_merge, df_bid_SP, left_on='Keyword suggestion', right_on='Keyword Text', how="outer",
                        indicator=True)
    df_merge = df_merge[df_merge['_merge'] == 'left_only']
    print(df_merge)
    return df_merge


def exact_match_suggestion():
    dataset = pd.read_excel('./data/raw/Sponsored Products Search term report.xlsx')
    data = pd.DataFrame(dataset[['Customer Search Term', 'Impressions', 'Clicks', 'Spend', '7 Day Total Sales ',
                                 '7 Day Total Orders (#)']])

    # data['Number of word'] = data['Customer Search Term'].str.count(' ') + 1
    # print(data)
    ## Drop orders less than 1
    new_data = data.drop(data[data['7 Day Total Orders (#)'] <= 1].index)
    print(new_data)
    df_bid_SP = input_output.read_bid('SP', path_bid)
    df_bid_SP = dataframe.select_row_by_val(df_bid_SP, 'Match Type', 'exact')
    df_merge = new_data.sort_values(by='7 Day Total Orders (#)', ascending=False)
    df_merge = pd.merge(df_merge, df_bid_SP, left_on='Customer Search Term', right_on='Keyword Text', how="outer",
                        indicator=True)
    df_merge = df_merge[df_merge['_merge'] == 'left_only']
    print(df_merge)
    return df_merge


def main():
    phrase_match_suggestion()
    exact_match_suggestion()
