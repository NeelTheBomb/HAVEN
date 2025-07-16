import argparse
import os
import sys
import pytaxonkit

sys.path.append(os.path.join(os.getcwd(), "src"))
from pathlib import Path
from data_preprocessing import dataset_parser, dataset_filter
from utils import external_sources_utils

UNIREF = "uniref"
UNIPROT = "uniprot"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocess the UniRef90 protein sequences dataset.\nOnly one of the below options can be selected at runtime.')
    parser.add_argument("-id", "--id_col", required=False,
                        help="Name of the id column. Example values: uniref50_id, uniref90_id, uniprot_id\n")
    parser.add_argument("-if", "--input_file", required=True,
                        help="Absolute path to input file depending on the option(s) selected.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory where the generated file will be saved.\n")
    parser.add_argument("--fasta_to_csv", action="store_true",
                        help="Convert the input fasta file to csv format.\n")
    parser.add_argument("-it", "--input_type",
                        help="Type of input file. Mandatory config option while converting from fasta to csv. Support values = 'uniref50', 'uniref90', 'uniref100', 'uniprot'\n")
    parser.add_argument("--uniprot_metadata", action="store_true",
                        help="Get metadata (hosts and embl reference id) of virus from UniProt.\n")
    parser.add_argument("--host_map_embl", action="store_true",
                        help="Get hosts of virus from EMBL.\n")
    parser.add_argument("--host_map_virushostdb",
                        help="Get hosts of virus from VirusHostDB mapping using the absolute path to the mapping file.\n")
    parser.add_argument("--prune_dataset", action="store_true",
                        help="Remove sequences without hosts of virus from the input csv dataset file.\n")
    parser.add_argument("--taxon_dir",
                        help="Absolute path to the NCBI taxon directory.")
    parser.add_argument("--taxon_metadata", action="store_true",
                        help="Get taxonomy metadata using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--uprank_host_genus", action="store_true",
                        help="Uprank the taxonomy of virus hosts to 'genus' level.")
    parser.add_argument("--taxon_kingdom", action="store_true",
                        help="Get kingdom taxonomy of virus hosts using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--taxon_class", action="store_true",
                        help="Get class taxonomy of virus hosts using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--filter_species_virus", action="store_true",
                        help="Filter for virus with rank of species.")
    parser.add_argument("--filter_species_virus_host", action="store_true",
                        help="Filter for virus hosts with rank of species.")

    # parser.add_argument("--filter_mammals_aves", action="store_true",
    #                     help="Filter for virus hosts belonging to mammalia OR aves family using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--filter_vertebrates", action="store_true",
                        help="Filter for virus hosts belonging to Vertebrata clade using the absolute path to the NCBI taxon directory provided in --taxon_dir.")
    parser.add_argument("--merge_sequence_data",
                        help="Join the metadata from the input_file with the sequence data from the provided absolute file path.")
    # parser.add_argument("--remove_multi_host_sequences", action="store_true",
    #                     help="Remove sequences with more than one host.")
    # parser.add_argument("--remove_single_host_viruses", action="store_true",
    #                     help="Remove viruses with only one host.")

    args = parser.parse_args()
    return args


def process(config):
    id_col = config.id_col
    input_file_path = config.input_file
    output_dir = config.output_dir

    # 2A. Metadata (host, embl ref id) from UniProt
    if config.uniprot_metadata:
        uniprot_metadata_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_uniprot_metadata.csv")
        print(f"uniprot_metadata_file_path ={uniprot_metadata_file_path}")
        query_func = external_sources_utils.query_uniref if UNIREF in config.input_type else external_sources_utils.query_uniprot
        dataset_filter.get_metadata_from_uniprot(input_file_path=input_file_path,
                                                 output_file_path=uniprot_metadata_file_path,
                                                 id_col=id_col,
                                                 query_uniprot=query_func,
                                                 input_type=config.input_type)
    # 2B. Host mapping from EMBL
    if config.host_map_embl:
        embl_host_mapping_filepath = os.path.join(output_dir, Path(input_file_path).stem + "_embl_host_mapping.csv")
        dataset_embl_hosts_mapping_filepath = os.path.join(output_dir, Path(input_file_path).stem + "_embl_hosts.csv")
        dataset_filter.get_virus_hosts_from_embl(input_file_path=input_file_path,
                                                 embl_mapping_filepath=os.path.join(output_dir,
                                                                                    embl_host_mapping_filepath),
                                                 output_file_path=os.path.join(output_dir,
                                                                               dataset_embl_hosts_mapping_filepath),
                                                 id_col=id_col)
    # 3. Remove sequences with no hosts
    if config.prune_dataset:
        pruned_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_pruned.csv")
        dataset_filter.remove_sequences_w_no_hosts(input_file_path=input_file_path,
                                                   output_file_path=pruned_dataset_file_path)

    # 4. Get taxonomy metadata (rank of virus and virus hosts) from NCBI
    if config.taxon_metadata:
        metadata_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_metadata.csv")
        dataset_filter.get_virus_metadata(input_file_path=input_file_path,
                                          taxon_metadata_dir_path=config.taxon_dir,
                                          output_file_path=metadata_dataset_file_path,
                                          id_col=id_col)

    if config.uprank_host_genus:
        upranked_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_virus_host_genus.csv")
        dataset_filter.uprank_virus_host_genus(input_file_path=input_file_path,
                                               taxon_metadata_dir_path=config.taxon_dir,
                                               output_file_path=upranked_dataset_file_path)

    # For dataset analysis
    if config.taxon_kingdom:
        kingdom_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_kingdom.csv")
        dataset_filter.get_virus_host_kingdom(input_file_path=input_file_path,
                                              taxon_metadata_dir_path=config.taxon_dir,
                                              output_file_path=kingdom_dataset_file_path)
    # For dataset analysis
    if config.taxon_class:
        class_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_class.csv")
        dataset_filter.get_virus_host_class(input_file_path=input_file_path,
                                            taxon_metadata_dir_path=config.taxon_dir,
                                            output_file_path=class_dataset_file_path)

    # 5. Filter for virus at species level
    if config.filter_species_virus:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_species_virus.csv")
        dataset_filter.get_virus_at_species_level(input_file_path=input_file_path,
                                                  output_file_path=filtered_dataset_file_path)

    # 6. Filter for virus_hosts at species level
    if config.filter_species_virus_host:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_species_virus_host.csv")
        dataset_filter.get_virus_host_at_species_level(input_file_path=input_file_path,
                                                       output_file_path=filtered_dataset_file_path)

    # 7. Filter for virus_hosts belonging to Vertebrata clade
    if config.filter_vertebrates:
        filtered_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_vertebrates.csv")
        dataset_filter.get_sequences_from_vertebrata_hosts(input_file_path=input_file_path,
                                                           taxon_metadata_dir_path=config.taxon_dir,
                                                           output_file_path=filtered_dataset_file_path)

    # 8. Merge the metadata with the sequence data
    if config.merge_sequence_data:
        sequence_dataset_file_path = os.path.join(output_dir, Path(input_file_path).stem + "_w_seq.csv")
        dataset_filter.join_metadata_with_sequences_data(input_file_path=input_file_path,
                                                         sequence_data_file_path=config.merge_sequence_data,
                                                         output_file_path=sequence_dataset_file_path,
                                                         id_col=id_col)


def pre_process(config):
    input_file_path = config.input_file
    output_dir = config.output_dir
    id_col = config.id_col

    df = None
    # 1. Parse the Fasta file
    if config.fasta_to_csv:
        if UNIREF in config.input_type:
            df = dataset_parser.parse_uniref_fasta_file(input_file_path=input_file_path, id_col=id_col)
        elif config.input_type == UNIPROT:
            df = dataset_parser.parse_uniprot_fasta_file(input_file_path=input_file_path, id_col=id_col)

        # write the parsed dataframe to a csv file
        output_file_path = os.path.join(output_dir, Path(input_file_path).stem + ".csv")
        print(f"Writing to file {output_file_path}")
        df.to_csv(output_file_path, index=False)


def main():
    config = parse_args()
    pre_process(config)
    process(config)
    return


if __name__ == "__main__":
    main()
    exit(0)
