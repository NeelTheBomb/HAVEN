# Util functions to query external sources  -
#  - PyTaxonKit: to get taxonomy
#  - UniProt: to get organism host and EMBL ids
#  - EMBL: to get organism host
import random

import requests
import pytaxonkit
import pandas as pd
import os
from Bio import SeqIO

# UniProt keywords/contsant values
UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb/search"
UNIREF_REST_API = "https://rest.uniprot.org/uniref/%s.json"
UNIREF100_QUERY_PARAM = "uniref_cluster_100:%s"
UNIREF90_QUERY_PARAM = "uniref_cluster_90:%s"
UNIREF50_QUERY_PARAM = "uniref_cluster_50:%s"

# EMBL keywords/constant values
EMBL_REST_API = "https://www.ebi.ac.uk/Tools/dbfetch"

# NCBI Taxonomy keywords
NAME = "Name"
RANK = "Rank"
NCBI_TAX_ID = "TaxID"
NCBI_Lineage = "Lineage"
TAXONKIT_DB = "TAXONKIT_DB"
MAMMALIA = "Mammalia"
AVES = "Aves"
VERTEBRATA_TAX_ID = "7742"


# query UniRef for to get the host of the virus of the protein sequence
# input: uniref_id
# output: list of host(s) of the virus
def query_uniref(uniref_id, input_type):
    query_param = None
    if input_type == "uniref100":
        query_param = UNIREF100_QUERY_PARAM
    elif input_type == "uniref90":
        query_param = UNIREF90_QUERY_PARAM
    elif input_type == "uniref50":
        query_param = UNIREF50_QUERY_PARAM
    else:
        print("ERROR: Invalid input type for UniRef dataset. Supported values are 'uniref100', 'uniref90', and, 'uniref50'")
        exit(1)

    response = requests.get(url=UNIPROT_REST_API,
                            params={"query": query_param % uniref_id,
                                    "fields": ",".join(["virus_hosts", "xref_embl"])})
    # respnse contains only the uniprot id of the sequences. Hence, we need to extract the uniprot id from the uniref_id only to parse the response.
    uniprot_id = uniref_id.split("_")[1]
    return parse_uniprot_response(response, uniprot_id)


# query Uniprot for to get the host of the virus of the protein sequence
# input: uniprot_id
# output: list of host(s) of the virus
def query_uniprot(uniprot_id, input_type):
    response = requests.get(url=UNIPROT_REST_API,
                            params={"query": uniprot_id,
                                    "fields": ",".join(["virus_hosts", "xref_embl"])})
    return parse_uniprot_response(response, uniprot_id)


def parse_uniprot_response(response, id):
    host_tax_ids = []
    embl_entry_id = None
    try:
        results = response.json()["results"]
        # ideally there should be only one matching primaryAccession entry for the seed uniprot_id
        data = [result for result in results if result["primaryAccession"] == id][0]

        # embl cross reference entry id
        cross_refs = data["uniProtKBCrossReferences"]
        embl_cross_ref_properties = \
            [cross_ref for cross_ref in cross_refs if cross_ref["database"] == "EMBL"][0]["properties"]
        embl_entry_id = \
            [property for property in embl_cross_ref_properties if property["key"] == "ProteinId"][0][
                "value"]

        # organism hosts from uniprot
        org_hosts = data["organismHosts"]
        for org_host in org_hosts:
            host_tax_ids.append(org_host["taxonId"])

    except (KeyError, IndexError):
        # to differentiate between the absence of mapping for a given sequence and
        # a sequence with mapping but zero hosts
        host_tax_ids = None
        pass
    return host_tax_ids, embl_entry_id


# Get taxonomy names and ranks from ncbi using pytaxonkit for given list of tax_ids
# Input: list of tax_ids
# Output: Dataframe with columns: ["TaxID", "Name", "Rank"]
def get_taxonomy_name_rank_from_id(tax_ids):
    # There is no method with input parameter: taxid and output: scientific name and rank.
    # However, there is a method that takes in name and returns the taxid and rank
    # Hack:
    # 1. Get names of the tax_ids using name()
    # 2. Get ranks using the names from previous step using name2taxid()
    df = pytaxonkit.name(tax_ids)
    df_w_rank = pytaxonkit.name2taxid(df[NAME].values)
    # default datatype of TaxID column = int32
    # convert it to int64 for convenience in downstream analysis
    df_w_rank[NCBI_TAX_ID] = pd.to_numeric(df_w_rank[NCBI_TAX_ID], errors="coerce").fillna(0).astype("int64")
    return df_w_rank


# Get taxonomy kingdom name from ncbi using pytaxonkit for given list of tax_ids
# Input: list of tax_ids
# Output: Dataframe with columns: ["TaxID", "kingdom"]
def get_taxonomy_kingdom_from_id(tax_ids):
    df = pytaxonkit.lineage(tax_ids, formatstr="{K}")
    # default datatype of TaxID column = int32
    # convert it to int64 for convenience in downstream analysis
    df[NCBI_TAX_ID] = pd.to_numeric(df[NCBI_TAX_ID], errors="coerce").fillna(0).astype("int64")
    # rename "Lineage" column to "Kingdom"
    df.rename(columns={NCBI_Lineage: "kingdom"}, inplace=True)
    return df


# Get taxonomy class name from ncbi using pytaxonkit for given list of tax_ids
# Input: list of tax_ids
# Output: Dataframe with columns: ["TaxID", "class"]
def get_taxonomy_class_from_id(tax_ids):
    df = pytaxonkit.lineage(tax_ids, formatstr="{c}")[[NCBI_TAX_ID, NCBI_Lineage]]
    # default datatype of TaxID column = int32
    # convert it to int64 for convenience in downstream analysis
    df[NCBI_TAX_ID] = pd.to_numeric(df[NCBI_TAX_ID], errors="coerce").fillna(0).astype("int64")
    # rename "Lineage" column to "Kingdom"
    df.rename(columns={NCBI_Lineage: "class"}, inplace=True)
    return df


# Get taxonomy names and ranks from ncbi using pytaxonkit for given list of tax_names
# Input: list of names
# Output: Dataframe with columns: ["TaxID", "Name", "Rank"]
def get_taxonomy_name_rank_from_name(tax_names):
    df_w_rank = pytaxonkit.name2taxid(tax_names)
    # default datatype of TaxID column = int32
    # convert it to int64 for convenience in downstream analysis
    df_w_rank[NCBI_TAX_ID] = pd.to_numeric(df_w_rank[NCBI_TAX_ID], errors="coerce").fillna(0).astype("int64")
    return df_w_rank


# Get taxids belonging to the clade = Vertebrata
# Input: list of tax_ids
# Output: list of tax_ids belonging to Vertebrata clade
def get_vertebrata_tax_ids(tax_ids):
    vertebrata_tax_ids = []
    for tax_id in tax_ids:
        # Issue: No placeholder formatter for rank=clade. Hence cannot use formatstr as for class {c}
        # Workaround: Get full lineage and filter for vertebrata Tax ID
        # example output from pytaxonkit.lineage([]):
        # '131567;2759;33154;33208;6072;33213;33511;7711;89593;7742;7776;117570;117571;8287;1338369;32523;32524;40674;32525;9347;1437010;314146;9443;376913;314293;9526;314295;9604;207598;9605;9606'
        # hence split by ";"
        try:
            full_lineage_tax_ids = pytaxonkit.lineage([tax_id])["FullLineageTaxIDs"].iloc[0].split(";")
            if VERTEBRATA_TAX_ID in full_lineage_tax_ids:
                vertebrata_tax_ids.append(tax_id)
        except:
            print(f"ERROR in lineage for tax_id = {tax_id}")
    return vertebrata_tax_ids


# For given tax_ids at rank lower than species, get the species equivalent ranks
# Input: tax ids at ranks lower than species
# Output: Taxonomy rank at species level
def get_taxonomy_species_data(tax_ids):
    lower_than_species_tax_ids = pytaxonkit.filter(tax_ids, lower_than="species")
    print(f"Tax ids with ranks less than species = {lower_than_species_tax_ids}")
    df_w_species_data = pytaxonkit.lineage(lower_than_species_tax_ids, formatstr="{s}")
    if df_w_species_data is None:
        return None
    df_w_species_data = df_w_species_data[[NCBI_TAX_ID, NAME, NCBI_Lineage]]
    tax_id_species_name_map = df_w_species_data.set_index(NCBI_TAX_ID)[NCBI_Lineage].to_dict()
    return tax_id_species_name_map


# For given tax_ids at rank lower than genus, get the genus equivalent ranks
# Input: tax ids at ranks lower than genus
# Output: Taxonomy rank at genus level
def get_taxonomy_genus_data(tax_ids):
    lower_than_genus_tax_ids = tax_ids
    # lower_than_genus_tax_ids = pytaxonkit.filter(tax_ids, lower_than="genus")
    print(f"Number of tax ids with ranks less than genus = {len(lower_than_genus_tax_ids)}")
    df_w_genus_data = pytaxonkit.lineage(lower_than_genus_tax_ids, formatstr="{g}")
    if df_w_genus_data is None:
        return None, None
    df_w_genus_data = df_w_genus_data[[NCBI_TAX_ID, NAME, NCBI_Lineage]]
    genus_tax_name_map = df_w_genus_data.set_index(NAME)[NCBI_Lineage].to_dict()
    print(f"Number of tax ids with genus equivalents = {len(genus_tax_name_map)}")
    print(genus_tax_name_map)
    return genus_tax_name_map


# Get taxids belonging to the class of mammals and aves
# Input: list of tax_ids
# Output: list of tax_ids belonging to mammals and aves class
def get_mammals_aves_tax_ids(tax_ids):
    mammals_aves_tax_ids = []
    for i, tax_id in enumerate(tax_ids):
        tax_class = pytaxonkit.lineage([tax_id], formatstr="{c}")[NCBI_Lineage].iloc[0]
        print(f"{i}: {tax_id} = {tax_class}")
        if tax_class == MAMMALIA or tax_class == AVES:
            mammals_aves_tax_ids.append(tax_id)
    return mammals_aves_tax_ids


# query EMBL for to get the host of the virus of the protein sequence
# input: embl_ref_id
# output: host name(s) of the virus
def query_embl(embl_ref_ids, temp_dir):
    response = requests.get(url=EMBL_REST_API, params=dict(format="embl", style="raw", id=",".join(embl_ref_ids)))
    # BioPython's SeqIO only takes input in the form of files.
    # Hack:
    # 1. Write the REST API's response to a temporary file to be processed using BioPython
    # 2. Capture the host
    # 3. Delete the temp file at the end of processing

    temp_output_file_path = os.path.join(temp_dir, "temp_" + str(random.randint(0, 1e9)) + ".txt")
    with open(temp_output_file_path, "w") as f:
        f.write(response.text)
    embl_host_mapping = {}
    index = 0
    try:
        # for record in wrapper(SeqIO.parse(temp_output_file_path, "embl")):
        for record in SeqIO.parse(temp_output_file_path, "embl"):
            index += 1

            # find the source feature which contains the host information
            source_feature = None
            for feature in record.features:
                if feature.type == "source":
                    source_feature = feature
                    break

            host = None
            try:
                if source_feature and source_feature.qualifiers["host"]:
                    host = source_feature.qualifiers["host"]
            except KeyError:
                pass
            embl_host_mapping[record.id] = host
    except ValueError as ve:
        # catch parsing errors due to invalid values in response from EMBL.
        # exclude the file
        print(ve)
        print(f"Removing Sequence with invalid response: Index={index}, Value = {embl_ref_ids[index]}")
        os.remove(temp_output_file_path)
        # hack to catch error stemming from the iterator itself.
        # drop the sequence having invalid response.
        # Recursion: return the host mapping you have so far and query for all remaining sequences.
        return embl_host_mapping | query_embl(embl_ref_ids[index + 1:], temp_dir)  # | is used to adding two dictionaries

    # delete the temporary file
    os.remove(temp_output_file_path)
    return embl_host_mapping


# query UniProt to get the cluster members of the UniRef cluster
# input: uniref_id
# output:
def get_uniref_cluster_members(uniref_id):
    response = requests.get(url=UNIREF_REST_API % uniref_id)
    member_ids = []

    if response.ok:
        data = response.json()
        member_count = data['memberCount']
        member_ids = [data["representativeMember"]["memberId"]]
        if member_count > 1:
            member_ids += [member["memberId"] for member in data["members"]]
    return member_ids
