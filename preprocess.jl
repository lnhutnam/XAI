using CSV
using DataFrames

const attr_file = "src_data/list_attr_celeba.txt"
custom_header = open(attr_file, "r") do io
    readline(io)                     # Skip the first line (sample count)
    header_line = readline(io)       # Read the header line (attributes)
    headers = split(header_line)     # Split into individual attribute names
    return String["Filename"; headers]  # Prepend "Filename" and ensure String type
end

const skipto_line = 3  # Data starts at line 3

# Read only the "Male" column:
df_male = DataFrame(CSV.File(attr_file;
    delim=' ', 
    ignorerepeated=true, 
    skipto=skipto_line,
    header=custom_header,
    select=["Male"]))

# Read only the "Filename" column:
df_filename = DataFrame(CSV.File(attr_file;
    delim=' ', 
    ignorerepeated=true, 
    skipto=skipto_line,
    header=custom_header,
    select=["Filename"]))

# Combine the filename and the Male column (they have the same number of rows)
df_attrs = hcat(df_filename, df_male)

# Replace -1 with 0 in the "Male" column (i.e. convert -1 to 0 for female)
df_attrs[!, "Male"] = replace(df_attrs[!, "Male"], -1 => 0)

const part_file = "src_data/list_eval_partition.txt"
df_part = DataFrame(CSV.File(part_file;
    header=false,
    delim=' ',
    ignorerepeated=true))

# Name the columns for the partition file
rename!(df_part, [:Filename, :Partition])

df_merged = innerjoin(df_attrs, df_part, on=:Filename)

const out_file = "data/celeba-gender-partitions.csv"
CSV.write(out_file, df_merged)

# Read back (if needed) ensuring missing strings are recognized
df_all = DataFrame(CSV.File(out_file, missingstring=["", "NA"]))

# Split the dataset according to partition values (0: train, 1: valid, 2: test)
CSV.write("data/celeba-gender-train.csv", filter(row -> row.Partition == 0, df_all))
CSV.write("data/celeba-gender-valid.csv", filter(row -> row.Partition == 1, df_all))
CSV.write("data/celeba-gender-test.csv",  filter(row -> row.Partition == 2, df_all))