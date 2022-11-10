from .utils import input_output, dataframe, amazon_campaign_link, campaign_management
import pandas as pd
from itertools import permutations


# import math
# 	Purpose: find keyword/PAT in one campaign which is not in another
#   KW: 	Manual KW SP  	<->		Manual KW SB 							: Campaign title/matchtype
#	PAT: 	Manual PAT SP  	<-> 	Manual PAT SB 	<-> 	Manual PAT SD   : Campaign title/matchtype
#	The destination can be in the same campaign or in different campaigns
#	0


def main():
    path = './data Fititude US/'
    try:
        linkdf = pd.read_excel(io=path + 'raw/link-manual-manual.xlsx', header=0, engine="openpyxl")
    except:
        amazon_campaign_link.campaign_link_manual(path)

    df_bid_SP = input_output.read_bid('SP', path)
    df_bid_SB = input_output.read_bid('SB', path)
    df_bid_SD = input_output.read_bid('SD', path)
    df_link = input_output.read_link(path)
    # print(dfadgrsource)
    try:
        campaign_management.delete_keyword_creation(path)
    except:
        print('no file to delete')

    for i, row in linkdf.iterrows():
        # adgrsource=dataframe.select_row_by_val(df_bid[0],'Entity','Ad Group','Campaign State (Informational only)','enabled','Ad Group State (Informational only)','enabled')
        if row['Campaign source type'] == 'Sponsored Products':
            dfsrc = df_bid_SP
        if row['Campaign source type'] == 'Sponsored Brands':
            dfsrc = df_bid_SB
        if row['Campaign source type'] == 'Sponsored Display':
            dfsrc = df_bid_SD
        if row['Campaign destination type'] == 'Sponsored Products':
            dfdst = df_bid_SP
        if row['Campaign destination type'] == 'Sponsored Brands':
            dfdst = df_bid_SB
        if row['Campaign destination type'] == 'Sponsored Display':
            dfdst = df_bid_SD
        df_adgr_src = dataframe.select_row_by_val(dfsrc, 'Ad Group Id', str(row['Ad Group source Id']), 'State',
                                                  'enabled')  # ,'Campaign State (Informational only)','enabled','Ad Group State (Informational only)','enabled')
        df_adgr_dest = dataframe.select_row_by_val(dfdst, 'Ad Group Id', str(row['Ad Group destination Id']), 'State',
                                                   'enabled')
        campaign_management.add_missing_keyword(df_adgr_src, df_adgr_dest, str(row['Ad Group source Id']),
                                                str(row['Ad Group destination Id']), path)


if __name__ == "__main__":
    main()
