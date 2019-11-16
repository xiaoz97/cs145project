import os
import sys
import zipfile


def getDataset(dataFolder=None):
	if dataFolder is None:
		dataFolder = os.path.normpath(os.path.dirname(os.path.realpath(__file__)) + "/../data")
		try:
			i = sys.argv.index('--data-folder')
			dataFolder = int(sys.argv[i + 1])
		except:
			pass

	if os.path.isfile(os.path.join(dataFolder, 'movies.csv')) is False:
		if os.system('kaggle competitions download -c uclacs145fall2019 -p "{0}"'.format(dataFolder)) != 0:
			print("Unable to download dataset through kaggle API. Did you install the API and configure your API key properly?", file=sys.stderr)
			print("Alternatively, you can specify the folder of the dataset with --data-folder.", file=sys.stderr)
			exit(1)
		else:
			with zipfile.ZipFile(os.path.join(dataFolder, 'uclacs145fall2019.zip'), 'r') as z:
				z.extractall(dataFolder)

	return dataFolder