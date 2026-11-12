"""Compare scalar loss and both gradients against one independent full projection."""
import argparse
import torch
from torch.nn import functional as F
from chunked_lm_loss import chunked_cross_entropy


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','mps'],required=True);a=p.parse_args()
    generator=torch.Generator().manual_seed(731)
    # Noncontiguous hidden input, uneven last chunk and a genuinely full head.
    hidden=torch.randn(2,7,11,generator=generator).transpose(0,1).to(a.device).requires_grad_()
    weight=torch.randn(97,11,generator=generator).to(a.device).requires_grad_()
    targets=torch.randint(97,(7,2),generator=generator).to(a.device)
    expected=F.cross_entropy(F.linear(hidden,weight).reshape(-1,97),targets.reshape(-1))
    gradients=torch.autograd.grad(expected,(hidden,weight))
    for chunk in [1,4,100]:
        actual=chunked_cross_entropy(hidden,weight,targets,chunk)
        torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
        for actual_grad,expected_grad in zip(torch.autograd.grad(actual,(hidden,weight)),gradients,strict=True):
            torch.testing.assert_close(actual_grad,expected_grad,atol=4e-6,rtol=2e-5)
    with torch.no_grad():
        torch.testing.assert_close(chunked_cross_entropy(hidden,weight,targets,4),expected,atol=3e-6,rtol=3e-6)
    print(f'{a.device}: full-vocabulary loss, hidden/head gradients, uneven chunks and no-grad match')


if __name__=='__main__':main()
