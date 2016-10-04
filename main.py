#Ayush: After learning, for 1000 epochs, the weight is from .62 to .77. Is that too large a variation?
import sys
import os
import argparse
import numpy as np
from factorgraph import *


Weight = np.dtype([("isFixed",      np.bool),
                   ("initialValue", np.float64)])

Variable = np.dtype([("isEvidence",   np.int8),
                     ("initialValue", np.int64),
                     ("dataType",     np.int16),
                     ("cardinality",  np.int64),
                     ("vtf_offset",   np.int64)])

Factor = np.dtype([("factorFunction", np.int16),
                   ("weightId",       np.int64),
                   ("featureValue",   np.float64),
                   ("arity",          np.int64),
                   ("ftv_offset",     np.int64)])

FactorToVar = np.dtype([("vid",            np.int64),
                        ("dense_equal_to", np.int64)])

VarToFactor = np.dtype([("value",               np.int64),
                        ("factor_index_offset",   np.int64),
                        ("factor_index_length", np.int64)])

def dataType(i):
    return {0: "Boolean",
            1: "Categorical"}.get(i, "Unknown")

def parse(argv):
  parser = argparse.ArgumentParser(description = 'A naive Gibbs Sampler')
  parser.add_argument('--dir', dest='directory', type=str, default='',
                    help = 'Directory containing factor graph files')
  parser.add_argument('--output_dir', dest='output_dir', type=str, default='.',
                      help = 'Output Directory')
  parser.add_argument('--l', dest='learning_epochs', type=int, default=0,
                    help = 'Number of learning epochs')
  parser.add_argument('--i', dest='inference_epochs', type=int, default=0,
                    help = 'Number of inference epochs')
  parser.add_argument('--b', dest='burn_in', type=int, default=0,
                    help = 'Number of burn in epochs')
  parser.add_argument('--info', dest='info', type=bool, default=False,
                    help = 'Print info')
  return parser.parse_args(argv)

def load_weights(data, nweights, weights):
  """TODO."""
  for i in range(nweights):
    # TODO: read types from struct?
    # TODO: byteswap only if system is little-endian
    buf = data[(17 * i):(17 * i + 8)]
    reverse_array(buf)
    weightId = np.frombuffer(buf, dtype=np.int64)[0]
    print ("weightID:")
    print(weightId)
    isFixed = data[17 * i + 8]
    buf = data[(17 * i + 9):(17 * i + 17)]
    reverse_array(buf)
    initialValue = np.frombuffer(buf, dtype=np.float64)[0]
    weights[weightId]["isFixed"] = isFixed
    weights[weightId]["initialValue"] = initialValue
  print("LOADED WEIGHTS")

def reverse_array(data):
  """TODO."""
  reverse(data, 0, data.size)

def reverse(data, start, end):
  """TODO."""
  end -= 1
  while (start < end):
    data[start], data[end] = data[end], data[start]
    start += 1
    end -= 1

def reverse_array(data):
    """TODO."""
    # TODO: why does this fail?
    # data = np.flipud(data)
    reverse(data, 0, data.size)


def load_variables(data, nvariables, variables):
    for i in range(nvariables):

        buf = data[(27 * i):(27 * i + 8)]
        reverse_array(buf)
        variableId = np.frombuffer(buf, dtype=np.int64)[0]

        isEvidence = data[27 * i + 8]

        buf = data[(27 * i + 9):(27 * i + 17)]
        reverse_array(buf)
        initialValue = np.frombuffer(buf, dtype=np.int64)[0]

        buf = data[(27 * i + 17):(27 * i + 19)]
        reverse_array(buf)
        dataType = np.frombuffer(buf, dtype=np.int16)[0]

        buf = data[(27 * i + 19):(27 * i + 27)]
        reverse_array(buf)
        cardinality = np.frombuffer(buf, dtype=np.int64)[0]

        variables[variableId]["isEvidence"] = isEvidence
        variables[variableId]["initialValue"] = initialValue
        variables[variableId]["dataType"] = dataType
        variables[variableId]["cardinality"] = cardinality

    print("LOADED VARS")


def load_factors(data, nfactors, factors, fmap, domain_mask, variable, vmap):
    """TODO."""
    index = 0
    fmap_idx = 0
    k = 0  # somehow numba 0.28 would raise LowerError without this line
    for i in range(nfactors):
        buf = data[index:(index + 2)]
        reverse_array(buf)
        factors[i]["factorFunction"] = np.frombuffer(buf, dtype=np.int16)[0]

        buf = data[(index + 2):(index + 10)]
        reverse_array(buf)
        arity = np.frombuffer(buf, dtype=np.int64)[0]
        factors[i]["arity"] = arity
        factors[i]["ftv_offset"] = fmap_idx

        index += 10  # TODO: update index once per loop?

        for k in range(arity):
            buf = data[index:(index + 8)]
            reverse_array(buf)
            vid = np.frombuffer(buf, dtype=np.int64)[0]
            fmap[fmap_idx + k]["vid"] = vid

            buf = data[(index + 8):(index + 16)]
            reverse_array(buf)
            val = np.frombuffer(buf, dtype=np.int64)[0]
            fmap[fmap_idx + k]["dense_equal_to"] = val
            index += 16
        fmap_idx += arity

        buf = data[index:(index + 8)]
        reverse_array(buf)
        factors[i]["weightId"] = np.frombuffer(buf, dtype=np.int64)[0]

        buf = data[(index + 8):(index + 16)]
        reverse_array(buf)
        factors[i]["featureValue"] = np.frombuffer(buf, dtype=np.float64)[0]
        # Ayush: Safe to ignore feature value.

        index += 16

    print("LOADED FACTORS")

def compute_var_map(variables, factors, fmap, vmap, factor_index, domain_mask):
    """TODO."""

    # Fill in factor_index and indexes into factor_index
    # Step 1: populate VTF.length
    # Ayush: there is one ftv per edge
    for ftv in fmap:
        vid = ftv["vid"]
        vtf = vmap[variables[vid]["vtf_offset"]]
        vtf["factor_index_length"] += 1

    # Step 2: populate VTF.offset
    last_len = 0
    last_off = 0
    for i, vtf in enumerate(vmap):
        vtf["factor_index_offset"] = last_off + last_len
        last_len = vtf["factor_index_length"]
        last_off = vtf["factor_index_offset"]

    # Step 3: populate factor_index
    offsets = vmap["factor_index_offset"].copy()
    for i, fac in enumerate(factors):
        for j in range(fac["ftv_offset"], fac["ftv_offset"] + fac["arity"]):
            ftv = fmap[j]
            vid = ftv["vid"]
            val = ftv["dense_equal_to"] if variables[
                vid]["dataType"] == 1 else 0
            vtf_idx = variables[vid]["vtf_offset"] + val
            fidx = offsets[vtf_idx]
            factor_index[fidx] = i
            offsets[vtf_idx] += 1

    # Step 4: remove dupes from factor_index
    for vtf in vmap:
        offset = vtf["factor_index_offset"]
        length = vtf["factor_index_length"]
        new_list = factor_index[offset: offset + length]
        new_list.sort()
        i = 0
        last_fid = -1
        for fid in new_list:
            if last_fid == fid:
                continue
            last_fid = fid
            factor_index[offset + i] = fid
            i += 1
        vtf["factor_index_length"] = i

"""
Loading Data
"""
parsed = parse(sys.argv[1:])
if parsed.info:
  print ("Parsed Data")
  print (parsed.directory)
  print (parsed.learning_epochs)
  print (parsed.training_epochs)
  print ()
meta = np.loadtxt(parsed.directory + '/' + 'graph.meta',
                  delimiter=',', dtype=int)
num_weights = meta[0]
num_variables = meta[1]
num_factors = meta[2]
num_edges = meta[3]

# load weights
weight_data = np.memmap(parsed.directory + '/graph.weights', mode='c')
weight = np.zeros(num_weights, Weight)
load_weights(weight_data, num_weights, weight)


if parsed.info:
  print("Weights:")
  for (i, w) in enumerate(weight):
    print("    weightId:", i)
    print("        isFixed:", w["isFixed"])
    print("        weight: ", w["initialValue"])
    print()

# load variables
variable_data = np.memmap(parsed.directory + '/graph.variables', mode='c')
variable = np.zeros(num_variables, Variable)
load_variables(variable_data, num_variables, variable)
sys.stdout.flush()
if parsed.info:
  print("Variables:")
  for (i, v) in enumerate(variable):
    print("    variableId:", i)
    print("        isEvidence:  ", v["isEvidence"])
    print("        initialValue:", v["initialValue"])
    print("        dataType:    ", v["dataType"],
         "(", dataType(v["dataType"]), ")")
    print("        cardinality: ", v["cardinality"])
    print()

# count total number of VTF records needed
num_vtfs = 0
for var in variable:
  var["vtf_offset"] = num_vtfs
  if var["dataType"] == 0:  # boolean
    num_vtfs += 1 # Ayush: Why?
  else:
    num_vtfs += var["cardinality"]
print("#VTF = %s" % num_vtfs)
sys.stdout.flush()

# generate variable-to-factor map
vmap = np.zeros(num_vtfs, VarToFactor)
factor_index = np.zeros(num_edges, np.int64)

sys.stdout.flush()

# load factors
factor_data = np.memmap(parsed.directory + "/graph.factors", mode="c")
factor = np.zeros(num_factors, Factor)
fmap = np.zeros(num_edges, FactorToVar)

load_factors(factor_data, num_factors,
             factor, fmap, None, variable, vmap)
sys.stdout.flush()

compute_var_map(variable, factor, fmap, vmap,
                factor_index, None)
print("COMPLETED VMAP INDEXING")
sys.stdout.flush()
fg = FactorGraph(weight, variable, factor, fmap, vmap, factor_index, fid=0)
"""
Learning
"""

def learning(burn_in, n_learning_epochs, stepsize, decay,
    regularization, reg_param, output_dir, fg):
  """TODO."""
  fg.learn(burn_in, n_learning_epochs,
           stepsize, decay, regularization, reg_param,
           learn_non_evidence=False)
  output_file = os.path.join(
        output_dir, "inference_result.out.weights.text")
  fg.dump_weights(output_file)

#fg.weight_value[0][0] = 0.663448541006

"""
Inference
"""
def inference(burn_in, n_inference_epoch, output_dir, fg):
  """TODO."""
  fg.inference(burn_in, n_inference_epoch, sample_evidence=True)
  output_file = os.path.join(output_dir, "inference_result.out.text")
  fg.dump_probabilities(output_file, n_inference_epoch)

learning(parsed.burn_in, parsed.learning_epochs, stepsize=0.01, decay=0.95,
    regularization=2, reg_param=.01, output_dir=parsed.output_dir, fg=fg)
inference(parsed.burn_in, parsed.inference_epochs, parsed.output_dir, fg)
