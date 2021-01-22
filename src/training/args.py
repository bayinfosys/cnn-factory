import os
import sys
import logging

import json
import argparse


logger = logging.getLogger(__name__)


DEFAULT_ARGS_SPEC="/src/train.args.spec.json"


def get_argument_parser():
  """
  set up the argparse parser by parsing a spec file

  env: ARGS_SPEC the location of the args spec file
       an array of json objects which define the arguments
       passed to argparse via the parser.add_argument method
       https://docs.python.org/3/library/argparse.html#the-add-argument-method
  """
  parser = argparse.ArgumentParser(
      description="train network on image/mask pairs"
  )

  if "ARGS_SPEC" not in os.environ:
    logger.info("ARGS_SPEC environment variable not set, using default value")
    os.environ["ARGS_SPEC"] = DEFAULT_ARGS_SPEC

  logger.info("parsing args.spec from '%s'" % os.environ["ARGS_SPEC"])

  # NOTE: FileNotFoundError intentionally causes termination
  with open(os.environ["ARGS_SPEC"], "r") as f:
    import builtins
    args_spec = json.load(f)
    for name, arg_spec in args_spec.items():
      logger.debug("argument: ['%s'] '%s" % (name, str(arg_spec)))
      # if the type is given, convert it to the callable function
      # from builtins (FIXME: what if it is a custom type?)
      if "type" in arg_spec:
        arg_spec["type"] = getattr(builtins, arg_spec["type"])

      parser.add_argument(name, **arg_spec)

  return parser


def parse_arguments(parser):
  """
  parse the process arguments from commandline or file

  env: ARGS_FILE location of a list of arguments which
       are loaded and passed directly to argparse via
       https://docs.python.org/3/library/argparse.html#the-parse-args-method

  if the environment variable is not defined, parameters
  are taken from the commandline as usual

  if the environment variable defines a file which cannot
  be opened, the exception is NOT caught here.
  """

  if "ARGS_FILE" in os.environ:
    # read the args from a json file
    logger.info("parsing '%s'" % os.environ["ARGS_FILE"])

    # NOTE: FileNotFoundError intentionally causes termination
    with open(os.environ["ARGS_FILE"], "r") as f:
      arguments = json.load(f)
  else:
    logger.info("parsing commandline")
    arguments = sys.argv

  logger.info("arguments: '%s'" % json.dumps(arguments))
  args = parser.parse_args(arguments)

  return args
