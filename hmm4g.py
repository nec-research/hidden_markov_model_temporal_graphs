"""
  Hidden Markov Model for Temporal Graphs

  File:     hmm4g.py
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
from typing import Tuple, Optional, List

import torch
from pydgn.model.interface import ModelInterface
from torch.nn.parameter import Parameter
from torch_geometric.data import Batch
from torch_scatter import scatter_add, scatter_max

from util import compute_unigram, compute_bigram


class HMM4G(ModelInterface):

    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        self.device = None

        self.readout_class = readout_class
        self.is_first_layer = config['depth'] == 1
        self.depth = config['depth']
        self.training = False
        self.return_node_embeddings = False

        self.K = dim_node_features
        self.Y = dim_target
        self.C = config['C']
        self.Cprev = config['C'] + 1
        self.unibigram = config['unibigram']

        self.readout = readout_class(dim_node_features, dim_edge_features,
                                     dim_target, config)

        self.eps = 1e-8  # used for Laplace smoothing

        if self.is_first_layer:
            # Define "prior" \pi_{i}, where i is state
            self.prior = Parameter(torch.empty([self.C], dtype=torch.float32), requires_grad=False)
            pr = torch.nn.init.uniform_(torch.empty(self.C))
            self.prior.data = pr / pr.sum()

            # Define "transition" A_{ij}, where i is the destination state and j the source state
            self.transition = Parameter(torch.empty([self.C, self.C], dtype=torch.float32), requires_grad=False)
            tr = torch.nn.init.uniform_(torch.empty(self.C, self.C))
            self.transition.data = tr / tr.sum(dim=0)  # given a src state j, normalize over possible dst states i

        else:
            # Define "prior" \pi_{ij'}, where i is state and j' the neighbor state
            self.prior = Parameter(torch.empty([self.C, self.Cprev], dtype=torch.float32), requires_grad=False)
            pr = torch.nn.init.uniform_(torch.empty(self.C, self.Cprev))
            self.prior.data = pr / pr.sum(dim=0)

            # Define "transition" A_{ijj'}, where i is the destination state and j the source state, j' is the neighbor state
            self.transition = Parameter(torch.empty([self.C, self.C, self.Cprev], dtype=torch.float32), requires_grad=False)
            tr = torch.nn.init.uniform_(torch.empty(self.C, self.C, self.Cprev))
            self.transition.data = tr / tr.sum(dim=0)  # given a src state j and an observable neighbor in state j', normalize over possible dst states i

        # Define suitable statistics accumulators for EM algorithm
        self.prior_numerator = Parameter(torch.empty_like(self.prior), requires_grad=False)
        self.transition_numerator = Parameter(torch.empty_like(self.transition), requires_grad=False)

        # Initialize the accumulators
        self.init_accumulators()

    def forward(self, data: Batch, prev_state=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        if not self.is_first_layer:
            x = torch.stack([snapshot[0].x for snapshot in data], dim=1)  # NxTxK
            edge_indices = [snapshot[0].edge_index for snapshot in data]
            batches = [snapshot[0].batch for snapshot in data]

            # Previous wrt layer
            prev_statistics = torch.stack([snapshot[1].stats for snapshot in data], dim=1)  # NxTxK

        else:
            x = torch.stack([snapshot.x for snapshot in data], dim=1)  # NxTxK
            edge_indices = [snapshot.edge_index for snapshot in data]
            batches = [snapshot.batch for snapshot in data]

            prev_statistics = None

        assert not torch.any(torch.isnan(x))

        # Previous wrt time
        prev_scaled_alpha = prev_state
        if prev_scaled_alpha is not None:
            prev_scaled_alpha.to(self.device)

        if self.is_first_layer:
            log_likelihood, scaled_alphas, scaling_coeffs, emissions, predictions = self.alpha_recursion_1(x, prev_scaled_alpha)
            scaled_betas = self.beta_recursion_1(emissions, scaling_coeffs)
            emission_posterior, transition_posterior = self.e_step_1(x, emissions, scaled_alphas, scaled_betas, scaling_coeffs)
            Q_EM = self.Q_EM_1(emissions, emission_posterior, transition_posterior)
        else:
            log_likelihood, scaled_alphas, scaling_coeffs, emissions, predictions, conditioned_priors, conditioned_transitions = self.alpha_recursion_l(x, prev_scaled_alpha, prev_statistics)
            scaled_betas = self.beta_recursion_l(emissions, conditioned_transitions, scaling_coeffs)
            emission_posterior, transition_posterior = self.e_step_l(x, emissions, conditioned_transitions, scaled_alphas, scaled_betas, scaling_coeffs)
            Q_EM = self.Q_EM_l(emissions, conditioned_priors, conditioned_transitions, emission_posterior, transition_posterior,
                               prev_alpha_was_none=prev_scaled_alpha is None)

        assert not torch.any(torch.isnan(scaled_alphas)), self.is_first_layer

        if self.return_node_embeddings:
            # Compute statistics for next layer

            # IMPORTANT NOTE:
            # Because we are predicting the next time step, we cannot use the posterior as unsupervised node embedding
            # as it will contain information about the subsequent time steps. When inferring with the model, in principle
            # the model makes a prediction for the next time step across all layers, and then it moves on to the next
            # time step. But because we are training one layer at a time, we need to first move across time and then
            # across layers. So, to store our "unsupervised" embeddings for subsequent classification of a model, we
            # have to used the alpha values, which are proportional to the posterior of the current state conditioned on the past
            assert not self.training


            if not self.is_first_layer:
                # Absorb dimension about Cprev, not needed anymore for computation of posterior
                sa = scaled_alphas.sum(-1)

                statistics_batch = self._compute_statistics(sa, edge_indices, batches, self.device)

                # Compute unigrams/unibigrams (using also the transition_posterior!)
                node_embedding_batch = self.compute_node_representations(sa, edge_indices, batches)

            else:
                statistics_batch = self._compute_statistics(scaled_alphas, edge_indices, batches, self.device)

                # Compute unigrams/unibigrams (using also the transition_posterior!)
                node_embedding_batch = self.compute_node_representations(scaled_alphas, edge_indices, batches)

            embeddings = (node_embedding_batch, scaled_alphas[:,-1,:], statistics_batch)
        else:
            embeddings = (None, scaled_alphas[:,-1,:])

        return predictions, embeddings, Q_EM, log_likelihood

    def alpha_recursion_1(self, x, prev_scaled_alpha):
        """
        Computes (scaled) alpha statistics and likelihood

        :param x: the sequence of node features, of dimension NxTxK
        :param prev_scaled_alpha: in case time-series are batched, this is the alpha_{t-1} to use instead of alpha_1
        :return: log_likelihood for each node, (scaled) alphas (NxTxC), list of scaling coefficients, emissions (NxTxC)
        """

        scaled_alphas = []
        scaling_coeffs = []
        emissions = []
        predictions = []

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            b = self.readout.p_x_given_Q(x_t)
            emissions.append(b)

            if t == 0:
                if prev_scaled_alpha is None:
                    alpha_t = b * self.prior.unsqueeze(0) # n x C
                else:
                    alpha_t = b * prev_scaled_alpha # n x C
            else:
                # see Bishop book, Eq 13.59
                alpha_t = b * \
                          torch.sum(self.transition.unsqueeze(0) *
                                    scaled_alphas[-1].unsqueeze(1), dim=-1) # n x C

            # see Bishop book, Eq 13.59
            c_t = alpha_t.sum(dim=1)  # normalize prob. of being in a particular state at time step t
            c_t[c_t == 0] = 1  # HARD FIX MAYBE THERE IS SOMETHING WRONG WITH THE ALPHA_T

            scaling_coeffs.append(c_t)

            scaled_alphas.append(alpha_t/c_t.unsqueeze(1))
            assert not torch.any(torch.isnan(scaled_alphas[-1]))

            emission_params = self.readout.get_emission_parameters()  # CxK

            # Eqs 23.2.11 and 23.2.39 Barber book
            prediction = (emission_params.unsqueeze(0).unsqueeze(2) * \
                         self.transition.unsqueeze(0).unsqueeze(3) * \
                         scaled_alphas[-1].unsqueeze(1).unsqueeze(3)).sum((1,2))  # NxK
            # for prediction, alpha should refer to the previous time step (see equation)
            # so, we add dimension 1 for timestep t to alpha, which implies that alpha refers to time t-1
            # similarly, the emission is unsqueezed wrt dim 2 (time t-1), because its dim 1 refers to time t
            predictions.append(prediction)

        scaled_alphas = torch.stack(scaled_alphas, dim=1)  # NxTxC
        scaling_coeffs = torch.stack(scaling_coeffs, dim=1)  # NxT
        emissions = torch.stack(emissions, dim=1) # NxTxC
        predictions = torch.stack(predictions, dim=1) # NxTxC

        # see Bishop book, Eq 13.63
        log_likelihood = torch.sum(scaling_coeffs.log(), dim=-1)

        # print(log_likelihood.shape, scaled_alphas.shape, scaling_coeffs.shape, emissions.shape, predictions.shape)

        return log_likelihood, scaled_alphas, scaling_coeffs, emissions, predictions

    def beta_recursion_1(self, emissions, scaling_coeffs):
        """
        Computes (scaled) beta statistics

        :param emissions: emissions for data points of dimension (NxTxC)
        :param scaling_coeffs: the scaling coefficients computed during the alpha recursion of dimension NxT

        :return: (scaled) betas (NxTxC)
        """

        scaled_betas = []

        beta_T = torch.ones(emissions.shape[0], emissions.shape[2]).to(emissions.device)
        scaled_betas.append(beta_T)

        for t in range(emissions.shape[1]):
            # at time step t, compute beta t-1
            if t == emissions.shape[1]-1:  # we cannot compute beta -1
                break

            b = emissions[:, -(t+1), :].unsqueeze(2)  # note the minus here, reversed order
            transition = self.transition.unsqueeze(0) # n x C_t x C_{t-1}
            scaled_beta_t = scaled_betas[-1].unsqueeze(2)  # append in reverse order of time. beta t+1

            # sum with respect to time t to obtain beta t minus 1
            beta_t_minus_1 = torch.sum(b * transition * scaled_beta_t, dim=1)
            scaled_betas.append(beta_t_minus_1/scaling_coeffs[:, -(t+1)].unsqueeze(1))  # note the minus here, reversed order

        # Bring scaled betas in order from t=0 to t=T
        scaled_betas.reverse()
        scaled_betas = torch.stack(scaled_betas, dim=1)  # NxTxC

        return scaled_betas

    def e_step_1(self, x, emissions, scaled_alphas, scaled_betas, scaling_coeff):
        """
        Compute statistics for e-step

        :param x: sequence of observations (NxTxK)
        :param emissions: emissions for data points of dimension (NxTxC)
        :param scaled_alphas: (scaled) alphas (NxTxC)
        :param scaled_betas: (scaled) betas (NxTxC)
        :param scaling_coeffs: the scaling coefficients computed during the alpha recursion of dimension NxT
        :return: emission_posterior and transition_posterior
        """
        emission_posterior = scaled_alphas * scaled_betas  # NxTxC

        # the last axis refers to time step t-1, the third to time step t
        transition_posterior = scaled_alphas[:, 0:-1, :].unsqueeze(2) * \
                               scaled_betas[:, 1:, :].unsqueeze(3) * \
                               emissions[:, 1:, :].unsqueeze(3) * \
                               self.transition.unsqueeze(0).unsqueeze(1) / \
                               scaling_coeff[:, 1:].unsqueeze(2).unsqueeze(3) # NxTxC_txC_{t-1}

        # TODO run checks that both posteriors are normalized!

        if self.training:
            # The readout is not concerned with time, so we can reshape the tensor as we had NxT samples
            self.readout._m_step(x.reshape(-1, self.K), emission_posterior.reshape(-1, self.C))
            self._m_step_1(emission_posterior, transition_posterior)

        return emission_posterior, transition_posterior

    def Q_EM_1(self, emissions, emission_posterior, transition_posterior):
        prior_Q_EM = (emission_posterior[:, 0, :] * self.prior.log().unsqueeze(0)).sum(1)
        emission_Q_EM = (emission_posterior * emissions.log()).sum((1,2))
        transition_Q_EM = (transition_posterior * self.transition.log().unsqueeze(0).unsqueeze(1)).sum((1,2,3))

        # EQ 23.3.2 Barber (recall: expectation of indicator variables coincide with probability)
        Q_EM = prior_Q_EM + transition_Q_EM + emission_Q_EM
        return Q_EM

    def _m_step_1(self, emission_posterior, transition_posterior):
        self.prior_numerator += emission_posterior.reshape(-1, self.C).mean(dim=0)
        self.transition_numerator += transition_posterior.reshape(-1, self.C, self.C).sum(0)

    def alpha_recursion_l(self, x, prev_scaled_alpha, prev_stats):
        """
        Computes (scaled) alpha statistics and likelihood

        :param x: the sequence of node features, of dimension NxTxK
        :param prev_scaled_alpha: in case time-series are batched, this is the alpha_{t-1} to use instead of alpha_1
        :param prev_stats: this represents information coming from the previous layer (NxTxCprev)
        :return: log_likelihood for each node, (scaled) alphas (NxTxC), list of scaling coefficients, emissions (NxTxC), conditional transition probs
        """

        # Compute the neighbourhood dimension for each vertex
        neighbDim = prev_stats.sum(dim=2, keepdim=True)  # --> ?N x T x 1
        # Replace zeros with ones to avoid divisions by zero
        # This does not alter learning: the numerator can still be zero
        neighbDim[neighbDim == 0] = 1.

        scaled_alphas = []
        scaling_coeffs = []
        emissions = []
        conditioned_priors = []
        conditioned_transitions = []
        predictions = []

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            b = self.readout.p_x_given_Q(x_t)
            emissions.append(b)

            stats_t = prev_stats[:, t, :]  # NxCprev
            neighbDim_t = neighbDim[:, t, :]  # Nx1
            norm_stats_t = (stats_t / neighbDim_t)  # NxCprev

            cond_p = self.prior.unsqueeze(0) * norm_stats_t.unsqueeze(1)  # NxCxCprev
            conditioned_priors.append(cond_p)

            cond_t = (self.transition.unsqueeze(0) * norm_stats_t.unsqueeze(1).unsqueeze(1))  # NxCxC_{t-1}xC_prev
            conditioned_transitions.append(cond_t)

            if t == 0:
                if prev_scaled_alpha is None:
                    alpha_t = b.unsqueeze(2) * (cond_p) # NxCxC_prev
                else:
                    alpha_t = b.unsqueeze(2) * prev_scaled_alpha # NxCxC_prev
            else:
                # see Bishop book, Eq 13.59 + our paper
                alpha_t = b.unsqueeze(2) * \
                          torch.sum(cond_t *  # NxCxC_{t-1}xC_prev
                                    scaled_alphas[-1].sum(-1).unsqueeze(1).unsqueeze(3), dim=-2) # NxCxC_prev
                # Note: we sum over the last dimension of alpha because it refers to the observable values
                # from the previous layer at the previous time step, so they are not interesting to compute the
                # posterior at timestep t-1.


            # see Bishop book, Eq 13.59
            c_t = alpha_t.sum(dim=1)  # normalize prob. of being in a particular state at time step t
            c_t[c_t == 0] = 1  # HARD FIX MAYBE THERE IS SOMETHING WRONG WITH THE ALPHA_T

            scaling_coeffs.append(c_t)

            scaled_alphas.append(alpha_t/c_t.unsqueeze(1))

            assert not torch.any(torch.isnan(scaled_alphas[-1])), (c_t)

            emission_params = self.readout.get_emission_parameters()  # CxK

            # Eqs 23.2.11 and 23.2.39 Barber book
            prediction = (emission_params.unsqueeze(0).unsqueeze(2) * \
                         cond_t.sum(-1).unsqueeze(3) * \
                         scaled_alphas[-1].sum(-1).unsqueeze(1).unsqueeze(3)).sum((1,2,3))  # NxK
            # for prediction, alpha should refer to the previous time step (see equation)
            # so, we add dimension 1 for timestep t to alpha, which implies that alpha refers to time t-1
            # similarly, the emission is unsqueezed wrt dim 2 (time t-1), because its dim 1 refers to time t
            # Also, see note above at line 301 about summing over the last dimension of alpha_{t-1}

            predictions.append(prediction)

        scaled_alphas = torch.stack(scaled_alphas, dim=1)  # NxTxC
        scaling_coeffs = torch.stack(scaling_coeffs, dim=1)  # NxT
        emissions = torch.stack(emissions, dim=1) # NxTxC
        predictions = torch.stack(predictions, dim=1) # NxTxC
        conditioned_priors = torch.stack(conditioned_priors, dim=1) # NxTxCxCprev
        conditioned_transitions = torch.stack(conditioned_transitions, dim=1) # NxTxCxC_{t-1}xCprev

        # see Bishop book, Eq 13.63
        log_likelihood = torch.sum(scaling_coeffs.log(), dim=(1,2))
        return log_likelihood, scaled_alphas, scaling_coeffs, emissions, predictions, conditioned_priors, conditioned_transitions

    def beta_recursion_l(self, emissions, conditioned_transitions, scaling_coeffs):
        """
        Computes (scaled) beta statistics

        :param emissions: emissions for data points of dimension (NxTxC)
        :param conditioned_transitions: conditioned transition for data points of dimension (NxTxCxCxCprev)
        :param scaling_coeffs: the scaling coefficients computed during the alpha recursion of dimension NxT
        :param prev_stats: this represents information coming from the previous layer

        :return: (scaled) betas (NxTxC)
        """

        scaled_betas = []

        beta_T = torch.ones(emissions.shape[0], emissions.shape[2], self.Cprev).to(emissions.device)
        scaled_betas.append(beta_T)

        for t in range(emissions.shape[1]):
            # at time step t, compute beta t-1
            if t == emissions.shape[1]-1:  # we cannot compute beta -1
                break

            b = emissions[:, -(t+1), :].unsqueeze(2).unsqueeze(3)  # NxCnote the minus here, reversed order
            conditioned_transition_t = conditioned_transitions[:, -(t+1), :, :, :]  # NxCxC_{t-1}xC_prev
            scaled_beta_t = scaled_betas[-1].sum(-1).unsqueeze(2).unsqueeze(3)  # append in reverse order of time. beta t+1
            # Also, see note above at line 301 about summing over the last dimension of alpha_{t-1} (here beta_{t+1})

            # print(b.shape, conditioned_transition_t.shape, scaled_betas[-1].shape, scaled_beta_t.shape)
            # print(scaling_coeffs[:, -(t+1)].shape); exit(0)

            # sum with respect to time t to obtain beta t minus 1
            beta_t_minus_1 = torch.sum(b * conditioned_transition_t * scaled_beta_t, dim=1)
            scaled_betas.append(beta_t_minus_1/scaling_coeffs[:, -(t+1)].unsqueeze(1))  # note the minus here, reversed order

        # Bring scaled betas in order from t=0 to t=T
        scaled_betas.reverse()
        scaled_betas = torch.stack(scaled_betas, dim=1)  # NxTxC

        return scaled_betas

    def e_step_l(self, x, emissions, conditioned_transitions, scaled_alphas, scaled_betas, scaling_coeff):
        """
        Compute statistics for e-step

        :param x: sequence of observations (NxTxK)
        :param emissions: emissions for data points of dimension (NxTxC)
        :param scaled_alphas: (scaled) alphas (NxTxC)
        :param scaled_betas: (scaled) betas (NxTxC)
        :param scaling_coeffs: the scaling coefficients computed during the alpha recursion of dimension NxT
        :param prev_stats: this represents information coming from the previous layer

        :return: emission_posterior and transition_posterior
        """
        emission_posterior = scaled_alphas * scaled_betas  # NxTxCxC_prev

        # print(emission_posterior.shape, scaled_alphas[:, 0:-1, :].sum(-1).unsqueeze(2).unsqueeze(4).shape);
        # print(scaled_betas[:, 1:, :].unsqueeze(3).shape, conditioned_transitions[:, 1:, :, :, :].shape);
        # print(scaling_coeff[:, 1:].unsqueeze(2).unsqueeze(3).shape); exit(0)

        # the last axis refers to time step t-1, the third to time step t
        # we should consider the observable variables Q_u at time step t, and sum over those at time step t-1
        # because that is what our learnable parameters need!
        transition_posterior = scaled_alphas[:, 0:-1, :].sum(-1).unsqueeze(2).unsqueeze(4) * \
                               scaled_betas[:, 1:, :].unsqueeze(3) * \
                               emissions[:, 1:, :].unsqueeze(3).unsqueeze(4) * \
                               conditioned_transitions[:, 1:, :, :, :] / \
                               scaling_coeff[:, 1:].unsqueeze(2).unsqueeze(3) # NxTxC_txC_{t-1}xCprev

        # TODO run checks that both posteriors are normalized!

        if self.training:
            # The readout is not concerned with time, so we can reshape the tensor as we had NxT samples
            self.readout._m_step(x.reshape(-1, self.K), emission_posterior.sum(-1).reshape(-1, self.C))
            self._m_step_l(emission_posterior, transition_posterior)

        return emission_posterior, transition_posterior

    def Q_EM_l(self, emissions, conditioned_priors, conditioned_transitions, emission_posterior, transition_posterior,
               prev_alpha_was_none):
        emission_Q_EM = (emission_posterior.sum(-1) * emissions.log()).sum((1,2))
        transition_Q_EM = (transition_posterior * conditioned_transitions[:,1:].log()).sum((1,2,3,4))

        # EQ 23.3.2 Barber (recall: expectation of indicator variables coincide with probability)
        Q_EM = transition_Q_EM + emission_Q_EM
        if prev_alpha_was_none:
            prior_Q_EM = (emission_posterior[:, 0, :] * conditioned_priors[:, 0, :].log()).sum((1,2))
            Q_EM += prior_Q_EM

        return Q_EM

    def _m_step_l(self, emission_posterior, transition_posterior):
        self.prior_numerator += emission_posterior.reshape(-1, self.C, self.Cprev).mean(dim=0)
        self.transition_numerator += transition_posterior.reshape(-1, self.C, self.C, self.Cprev).sum(0)


    def m_step(self):
        self.prior.data = self.prior_numerator / self.prior_numerator.sum(0)
        self.transition.data = self.transition_numerator / self.transition_numerator.sum(0)
        self.readout.m_step()

        self.init_accumulators()

    def _compute_statistics(self, scaled_alphas, edge_indices, batches, device):
        all_stats = []

        # Compute one set of statistics for each time step
        for t in range(len(edge_indices)):
            edge_index = edge_indices[t]
            batch = batches[t]

            statistics = torch.full((scaled_alphas.shape[0], scaled_alphas.shape[2] + 1), 0.,
                                    dtype=torch.float32).to(device)

            sparse_adj_matr = torch.sparse_coo_tensor(edge_index, \
                                                      torch.ones(edge_index.shape[1],
                                                                 dtype=scaled_alphas.dtype).to(device), \
                                                      torch.Size([scaled_alphas.shape[0],
                                                                  scaled_alphas.shape[0]])).to(device).transpose(0, 1)
            statistics[:, :-1] = torch.sparse.mm(sparse_adj_matr, scaled_alphas[:, t, :])

            # Deal with nodes with degree 0: add a single fake neighbor with uniform posterior
            degrees = statistics[:, :-1].sum(dim=[1]).floor()
            statistics[degrees == 0., :] = 1. / self.Cprev

            # use bottom states (all in self.Cprev-1)
            max_arieties, _ = self._compute_max_ariety(degrees.int().to(self.device), batch)
            max_arieties[max_arieties == 0] = 1
            statistics[:, self.C] += degrees / max_arieties[batch].float()

            assert not torch.any(torch.isnan(statistics))

            all_stats.append(statistics)

        return torch.stack(all_stats, dim=1)  # add time dimension --> NxTxCprev

    def compute_node_representations(self, scaled_alphas, edge_indices, batches):
        all_node_representations = []

        for t in range(len(edge_indices)):

            node_unigram = compute_unigram(scaled_alphas[:, t, :], True)

            if self.unibigram:
                node_bigram = compute_bigram(scaled_alphas[:, t, :], edge_indices[t], batches[t], True)

                node_embeddings_batch = torch.cat((node_unigram, node_bigram), dim=1)
            else:
                node_embeddings_batch = node_unigram

            assert not torch.any(torch.isnan(node_embeddings_batch))

            all_node_representations.append(node_embeddings_batch)

        return torch.stack(all_node_representations, dim=1)  # add time dimension --> NxTxC OR NxTx(C + C^2)

    def viterbi(self, snapshots):
        """
        Most likely sequence of hidden states, i.e., node representations for each time step

        :param snapshots: the sequence of node features, of dimension NxTxK
        :return:
        """
        # We don't need Viterbi! We want to propagate the posterior value given the sequence to the neighbors!
        pass

    def init_accumulators(self):
        self.readout.init_accumulators()

        torch.nn.init.constant_(self.prior_numerator, self.eps)
        torch.nn.init.constant_(self.transition_numerator, self.eps)

        # Do not delete this!
        if self.device:  # set by to() method
            self.to(self.device)

    def to(self, device):
        super().to(device)
        self.device = device

    def _compute_sizes(self, batch, device):
        return scatter_add(torch.ones(len(batch), dtype=torch.int).to(device), batch)

    def _compute_max_ariety(self, degrees, batch):
        return scatter_max(degrees, batch)
