import csv
import os
import pandas as pd

# Merge Google Series into one csv file

def get_all_files(directory, extension='.csv'):
    dir_list = os.listdir(directory)
    csv_files = []
    for i in dir_list:
        if i.endswith(extension):
            csv_files.append(os.path.realpath(i))
    return csv_files


csv_files = get_all_files(os.getcwd())

example_file = csv_files[0]


with open(example_file) as file:
	reader = csv.reader(file)
	rows = [r for r in reader]
	final_rows = rows[3:]

	months = [row[0] for row in final_rows]

	master_df = pd.DataFrame(
		{
		'col1': months,
		})

	master_df.columns = ['month']
	print(master_df.head())	


for example_file in csv_files:
	with open(example_file) as file:
		reader = csv.reader(file)
		rows = [r for r in reader]
		header = rows[0][0]
		header_splitted = header.split()
		header_splitted = header_splitted[1:]
		header_splitted = [''.join(header_splitted[0:])]
		header_splitted = header_splitted[0]

		final_rows = rows[2:]
		final_rows[0][1] = header_splitted

		final_rows = rows[3:]

		values = [row[1] for row in final_rows]

		print(values)

		series = pd.Series(values)
		master_df[header_splitted] = series




print(master_df.shape)
master_df.to_csv("google_indicators.csv", index=0 )
