import torch
import opt_einsum as oe

import factorizer.tensor_network as tn

#%%
tucker = tn.Tucker((3, 4, 5), rank=(2, 2, 2))
tucker = tucker.get_subnet(("U_1", "U_2", "G"))
tucker.output_edges


#%%
inputs = "ab,bc,cd,de,ef"
factors = [torch.rand((1000, 1000), requires_grad=True) for _ in range(5)]

with oe.sharing.shared_intermediates() as cache:  # create a cache
    pass
marginals = {}
for output in "abcdef":
    with oe.sharing.shared_intermediates(cache):  # reuse a common cache
        marginals[output] = oe.contract(f"{inputs}->{output}", *factors)
del cache  # garbage collect intermediates


#%%
cpd = tn.CanonicalPolyadic((3, 4, 5), rank=2, batch=False)
factors = {n: torch.rand(cpd.nodes[n]["shape"]) for n in cpd.nodes}
expr = cpd.contract_expression()[0]

with oe.sharing.shared_intermediates() as cache:  # create a cache
    pass

with oe.sharing.shared_intermediates(cache):  # reuse a common cache
    results = expr(factors)
