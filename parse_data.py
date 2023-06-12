import pysam
import os
import pandas as pd

def sanity(path): # print first 10 rows of the BAM file
    # Index the BAM or SAM file
    if not os.path.exists(path + '.bai'):
        pysam.index(path)
    # Open the BAM or SAM file
    samfile = pysam.AlignmentFile(path, "rb")  
    
    # Iterate over the first 10 reads
    for i, read in enumerate(samfile.fetch()):
        if i == 10:
            break
        print(f"Read {i+1}: {read.query_sequence}")

    # Close the file
    samfile.close()


def split_seq_to_words(bamfil_path):
    # Index the BAM or SAM file
    if not os.path.exists(bamfil_path + '.bai'):
        pysam.index(bamfil_path)
    # Open the BAM or SAM file
    samfile = pysam.AlignmentFile(bamfil_path, "rb")   
    splited_sequences = []
    for i, read in enumerate(samfile.fetch()):
        # if i == 10:
        #     break
        seq = read.query_sequence
        seq_lst = []
        sub_seq_lst = []
        for j in range(len(seq)):
            sub_seq_lst.append(seq[j])
            if ((j+1) % 6) == 0: # j range is 0 to 150
                sub_seq = "".join(sub_seq_lst)
                seq_lst.append(sub_seq)
                sub_seq_lst = []
        splited_sequences.append(" ".join(seq_lst))
        
    # Verify data is splited properly:
    for i, read in enumerate(splited_sequences):
        if i == 10:
            break
    # Tests
    #     print(f"Read {i+1}: {splited_sequences[i]}")
    # print(f"SEQ NUM IS: {len(splited_sequences)}")

    # Close the file
    samfile.close()
    return splited_sequences



def generate_data_frame_from_sample(sample_path, labels_dict):
    """
    params:
    sample_path: path to BAM file with genomic data of a patient
    labels_dict: labels dictionary we want to check according to metadata
    """
    splited_sequences = split_seq_to_words(sample_path)
    n = len(splited_sequences)
    for l in labels_dict:
        labels_dict[l] = [labels_dict[l]]*n
    df_dict = labels_dict
    df_dict["sequence"] = splited_sequences
    df = pd.DataFrame(df_dict)
    # TEST:
    # df.to_csv(f"{sample_path}.csv", sep='\t', index=False)
    return df

def build_tsv_file_from_dataframes(metadata:dict):
    """
    metadata: dictionary of samples to labels_dictionary
    """

    # Extract paths from metadata
    sample_paths = list(metadata.keys())
    # Generate df from one of the samples
    if len(sample_paths) > 0:
        path = sample_paths[0]
        df = generate_data_frame_from_sample(path, metadata[path])
    else:
        print("Please provide path for at least one sample")
    # Concat with the rest
    if len(sample_paths) > 1:
        for i in range(1, len(sample_paths)):
            path = sample_paths[i]
            df_to_concat = generate_data_frame_from_sample(path, metadata[path])
            # Test
            # print(df_to_concat)
            df = pd.concat([df, df_to_concat], ignore_index=True)
    # Generate tsv file with the processed & labeled data
    df.to_csv(f"data.tsv", sep='\t', index=False)

    
# sanity("1000247.bam")
# split_seq_to_words("1000247.bam")
# generate_data_frame_from_sample("1000247.bam", {"Sex":"Male"})
build_tsv_file_from_dataframes({"1000247.bam":{"label":"1"}, "1000273.bam":{"label":"0"}})