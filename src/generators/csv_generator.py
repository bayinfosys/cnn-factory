"""
functions for reading csv data into generators
"""
import logging


logger = logging.getLogger(__name__)


def csv_to_lists(csv_filename):
  """
  convert a csv to a map of lists
  key: column header in the csv file
  values: list of values for that column
  """
  import csv

  logger.info("reading '%s'" % csv_filename)

  with open(csv_filename, "r") as csv_f:
    reader = csv.reader(csv_f)
    header = next(reader)

    logger.debug("header: '%s'" % str(header))

    # create a dict of empty lists
    data = {k:[] for k in header}

    for row in reader:
      for idx, h in enumerate(header):
        data[h].append(row[idx])

  # check each row is the same length
  assert all([len(v) == len(list(data.values())[0]) for v in data.values()]), "inconsistent row lengths for all attributes"

  logger.info("read %i/%i rows/columns" % (len(list(data.values())[0]), len(data.keys())))

  return data
