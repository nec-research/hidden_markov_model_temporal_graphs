"""
  Hidden Markov Model for Temporal Graphs

  File:     emission.py
  Authors:  Federico Errica (federico.errica@neclab.eu)
	    Alessio Gravina (alessio.gravina@phd.unipi.it)
	    Davide Bacciu (davide.bacciu@unipi.it)
	    Alessio Micheli (alessio.micheli@unipi.it)
            <NAME OF B (email B)>

NEC Laboratories Europe GmbH, Copyright (c) 2023, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.


"""
import math

import scipy
import scipy.cluster
import scipy.cluster.vq
import torch
# Interface for all emission distributions
from torch.nn import ModuleList


class EmissionDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def init_accumulators(self):
        raise NotImplementedError()

    def e_step(self, x_labels, y_labels):
        raise NotImplementedError()

    def infer(self, p_Q, x_labels):
        raise NotImplementedError()

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        raise NotImplementedError()

    def m_step(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


# do not replace replace with torch.distributions yet, it allows GPU computation
class IsotropicGaussian(EmissionDistribution):

    def __init__(self, dim_features, dim_hidden_states, var_threshold=1e-3):
        super().__init__()

        self.eps = 1e-8  # Laplace smoothing
        self.var_threshold = var_threshold  # do not go below this value

        self.F = dim_features
        self.C = dim_hidden_states  # clusters

        self.mu = torch.nn.Parameter(torch.rand((self.C, self.F),
                                                dtype=torch.float32),
                                     requires_grad=False)
        self.var = torch.nn.Parameter(torch.rand((self.C, self.F),
                                                 dtype=torch.float32),
                                      requires_grad=False)
        self.pi = torch.nn.Parameter(torch.FloatTensor([math.pi]),
                                     requires_grad=False)

        self.mu_numerator = torch.nn.Parameter(torch.empty([self.C, self.F],
                                                           dtype=torch.float32),
                                               requires_grad=False)
        self.mu_denominator = torch.nn.Parameter(torch.empty([self.C, 1],
                                                             dtype=torch.float32),
                                                 requires_grad=False)
        self.var_numerator = torch.nn.Parameter(torch.empty([self.C, self.F],
                                                            dtype=torch.float32),
                                                requires_grad=False)
        self.var_denominator = torch.nn.Parameter(torch.empty([self.C, 1],
                                                              dtype=torch.float32),
                                                  requires_grad=False)

        # To launch k-means the first time
        self.initialized = False

        self.init_accumulators()

    def to(self, device):
        super().to(device)
        self.device = device

    def initialize(self, labels):
        codes, distortion = scipy.cluster.vq.kmeans(labels.cpu().detach().numpy()[:],
                                                    self.C, iter=20,
                                                    thresh=1e-05)
        # Number of prototypes can be < than self.C
        self.mu[:codes.shape[0], :] = torch.from_numpy(codes)
        self.var[:, :] = torch.std(labels, dim=0)

        self.mu = self.mu  # .to(self.device)
        self.var = self.var  # .to(self.device)

    def get_parameters(self):
        return self.mu

    def univariate_pdf(self, labels, mean, var):
        """
        Univariate case, computes probability distribution for each data point
        :param labels:
        :param mean:
        :param var:
        :return:
        """
        return torch.exp(-((labels.float() - mean) ** 2) / (2 * var)) / (torch.sqrt(2 * self.pi * var))

    def multivariate_diagonal_pdf(self, labels, mean, var):
        """
        Multivariate case, DIAGONAL cov. matrix. Computes probability distribution for each data point
        :param labels:
        :param mean:
        :param var:
        :return:
        """
        diff = (labels.float() - mean)

        log_normaliser = -0.5 * (torch.log(2 * self.pi) + torch.log(var))
        log_num = - (diff * diff) / (2 * var)
        log_probs = torch.sum(log_num + log_normaliser, dim=1)
        probs = torch.exp(log_probs)

        # Trick to avoid instability, in case variance collapses to 0
        probs[probs != probs] = self.eps
        probs[probs < self.eps] = self.eps

        return probs

    def init_accumulators(self):
        """
        This method initializes the accumulators for the EM algorithm.
        EM updates the parameters in batch, but needs to accumulate statistics in mini-batch style.
        :return:
        """
        torch.nn.init.constant_(self.mu_numerator, self.eps)
        torch.nn.init.constant_(self.mu_denominator, self.eps * self.C)
        torch.nn.init.constant_(self.var_numerator, self.eps)
        torch.nn.init.constant_(self.var_denominator, self.eps * self.C)

    def e_step(self, x_labels, y_labels):
        """
        For each cluster i, returns the probability associated to a specific label.
        :param x_labels: unused
        :param y_labels: output observables
        :return: a distribution associated to each layer
        """
        if not self.initialized:
            self.initialized = True
            self.initialize(y_labels)

        emission_of_labels = None
        for i in range(0, self.C):
            if emission_of_labels is None:
                emission_of_labels = torch.reshape(self.multivariate_diagonal_pdf(y_labels, self.mu[i], self.var[i]),
                                                   (-1, 1))
            else:
                emission_of_labels = torch.cat((emission_of_labels,
                                                torch.reshape(
                                                    self.multivariate_diagonal_pdf(y_labels, self.mu[i], self.var[i]),
                                                    (-1, 1))), dim=1)
        emission_of_labels += self.eps
        assert not torch.isnan(emission_of_labels).any(), (torch.sum(torch.isnan(emission_of_labels)))
        return emission_of_labels.detach()

    def infer(self, p_Q, x_labels):
        """
        Compute probability of a label given the probability P(Q) as argmax_y \sum_i P(y|Q=i)P(Q=i)
        :param p_Q: tensor of size ?xC
        :param x_labels: unused
        :return:
        """
        # We simply compute P(y|x) = \sum_i P(y|Q=i)P(Q=i|x) for each node
        inferred_y = torch.mm(p_Q, self.mu)  # ?xF
        return inferred_y

    def _m_step(self, x_labels, y_labels, posterior_estimate):
        """
        Updates the minibatch accumulators
        :param x_labels: unused
        :param y_labels: output observable
        :param posterior_estimate: a ?xC posterior estimate obtained using the output observables
        """
        y_labels = y_labels.float()

        for i in range(0, self.C):
            reshaped_posterior = torch.reshape(posterior_estimate[:, i], (-1, 1))  # for broadcasting with F > 1

            den = torch.unsqueeze(torch.sum(posterior_estimate[:, i], dim=0), dim=-1)  # size C

            y_weighted = torch.mul(y_labels, reshaped_posterior)  # ?xF x ?x1 --> ?xF

            y_minus_mu_squared_tmp = y_labels - self.mu[i, :]
            # DIAGONAL COV MATRIX
            y_minus_mu_squared = torch.mul(y_minus_mu_squared_tmp, y_minus_mu_squared_tmp)

            self.mu_numerator[i, :] += torch.sum(y_weighted, dim=0)
            self.var_numerator[i] += torch.sum(torch.mul(y_minus_mu_squared, reshaped_posterior), dim=0)
            self.mu_denominator[i, :] += den
            self.var_denominator[i, :] += den

    def m_step(self):
        """
        Updates the emission parameters and re-initializes the accumulators.
        :return:
        """
        self.mu.data = self.mu_numerator / self.mu_denominator
        # Laplace smoothing
        self.var.data = (self.var_numerator + self.eps) / (self.var_denominator + self.C * self.eps) + self.var_threshold

        self.init_accumulators()

    def __str__(self):
        return f"{str(self.mu)}, {str(self.mu)}"
