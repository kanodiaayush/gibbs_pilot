"""TODO."""

import numpy as np
import math


def gibbsthread(weight, variable,
        factor, fmap, vmap, factor_index, Z, cstart,
        count, var_value, weight_value, sample_evidence, burnin):
  """TODO."""
  # Indentify start and end variable
  nvar = variable.shape[0]
  start = 0 
  end = nvar
  for var_samp in range(start, end):
    if variable[var_samp]["isEvidence"] == 0 or sample_evidence:
      v = draw_sample(var_samp, weight, variable,
              factor, fmap, vmap, factor_index, Z[0],
              var_value, weight_value)
      var_value[0][var_samp] = v
      if not burnin:
        count[cstart[var_samp]] += v


def draw_sample(var_samp, weight, variable, factor,
        fmap, vmap, factor_index, Z, var_value, weight_value):
  """TODO."""
  cardinality = variable[var_samp]["cardinality"]
  for value in range(cardinality):
    Z[value] = np.exp(potential(var_samp, value,
                                weight, variable, factor, fmap,
                                vmap, factor_index, var_value,
                                weight_value))

  for j in range(1, cardinality):
    Z[j] += Z[j - 1]

  z = np.random.rand() * Z[cardinality - 1]

  # TODO: this looks at the full vector, slow if one var has high cardinality
  return np.argmax(Z >= z)


def potential(var_samp, value, weight, variable, factor,
              fmap, vmap, factor_index, var_value, weight_value):
  """TODO."""
  p = 0.0
  varval_off = 0 
  vtf = vmap[variable[var_samp]["vtf_offset"] + varval_off]
  start = vtf["factor_index_offset"]
  end = start + vtf["factor_index_length"]
  for k in range(start, end):
    factor_id = factor_index[k]
    p += weight_value[0][factor[factor_id]["weightId"]] * \
      factor[factor_id]["featureValue"] * \
      eval_factor(factor_id, var_samp, value, variable,
            factor, fmap, var_value)
  return p


FACTORS = {  # Factor functions for boolean variables
  "FUNC_IMPLY_NATURAL": 0,
  "FUNC_OR": 1,
  "FUNC_EQUAL": 3,
  "FUNC_AND": 2,
  "FUNC_ISTRUE": 4,
  "FUNC_LINEAR": 7,
  "FUNC_RATIO": 8,
  "FUNC_LOGICAL": 9,
  "FUNC_IMPLY_MLN": 13,
}

for (key, value) in FACTORS.items():
  exec(key + " = " + str(value))


def eval_factor(factor_id, var_samp, value, variable, factor, fmap,
        var_value):
  """TODO."""
  # Implementation of factor functions for categorical variables
  fac = factor[factor_id]
  ftv_start = fac["ftv_offset"]
  ftv_end = ftv_start + fac["arity"]

  if factor[factor_id]["factorFunction"] == FUNC_ISTRUE:
    for l in range(ftv_start, ftv_end):
      v = value if (fmap[l]["vid"] == var_samp) \
        else var_value[0][fmap[l]["vid"]]
      if v == 0:
        return -1
    return 1
  else:  # FUNC_UNDEFINED
    print("Error: Factor Function", factor[factor_id]["factorFunction"],
              "( used in factor", factor_id, ") is not implemented.")
    raise NotImplementedError("Factor function is not implemented.")
