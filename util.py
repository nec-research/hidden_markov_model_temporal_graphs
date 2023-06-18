"""
  Hidden Markov Model for Temporal Graphs

  File:     util.py
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
from typing import Optional, Tuple, List

import torch
import torch_geometric


def extend_lists(data_list: Optional[Tuple[Optional[List[torch.Tensor]]]],
                 new_data_list: Tuple[Optional[List[torch.Tensor]]]) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Extends the semantic of Python :func:`extend()` over lists to tuples
    Used e.g., to concatenate results of mini-batches in incremental architectures such as :obj:`CGMM`
    Args:
        data_list: tuple of lists, or ``None`` if there is no list to extend.
        new_data_list: object of the same form of :obj:`data_list` that has to be concatenated
    Returns:
        the tuple of extended lists
    """
    if data_list is None:
        return new_data_list

    assert len(data_list) == len(new_data_list)

    for i in range(len(data_list)):
        if new_data_list[i] is not None:
            data_list[i].extend(new_data_list[i])

    return data_list


def to_tensor_lists_temporal(embeddings: Tuple[Optional[torch.Tensor]]) -> Tuple[Optional[List[torch.Tensor]]]:
    r"""
    Reverts batched outputs back to a list of Tensors elements. Here the batch refers to time steps!

    Args:
        embeddings (tuple): a tuple of embeddings :obj:`(hidden_states, scaled_alphas, statistics)`.
                            Each embedding is a :class:`torch.Tensor`.
        batch (:class:`torch_geometric.data.batch.Batch`): Batch information used to split the tensors.
    Returns:
        a tuple with the same semantics as the argument ``embeddings``, but this time each element holds a list of
        Tensors, one for each time step in the single graph dataset.
    """
    node_embeddings, scaled_alphas, stats = embeddings

    node_embeddings = node_embeddings.detach() if node_embeddings is not None else None
    node_embeddings_list = [] if node_embeddings is not None else None

    stats = stats.detach() if stats is not None else None
    stats_list = [] if stats is not None else None

    assert node_embeddings.shape[1] == stats.shape[1]
    num_timesteps = node_embeddings.shape[1]

    for i in range(num_timesteps):
        node_embeddings_list.append(node_embeddings[:, i])
        stats_list.append(stats[:, i])

    return node_embeddings_list, stats_list


def compute_unigram(posteriors: torch.Tensor, use_continuous_states: bool) -> torch.Tensor:
    r"""
    Computes the unigram representation of nodes as defined in https://www.jmlr.org/papers/volume21/19-470/19-470.pdf
    Args:
        posteriors (torch.Tensor): tensor of posterior distributions of nodes with shape `(#nodes,num_latent_states)`
        use_continuous_states (bool): whether or not to use the most probable state (one-hot vector) or a "soft" version
    Returns:
        a tensor of unigrams with shape `(#nodes,num_latent_states)`
    """
    num_latent_states = posteriors.shape[1]

    if use_continuous_states:
        node_embeddings_batch = posteriors
    else:
        node_embeddings_batch = make_one_hot(posteriors.argmax(dim=1), num_latent_states)

    return node_embeddings_batch.float()


def compute_bigram(posteriors: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                   use_continuous_states: bool) -> torch.Tensor:
    r"""
    Computes the bigram representation of nodes as defined in https://www.jmlr.org/papers/volume21/19-470/19-470.pdf
    Args:
        posteriors (torch.Tensor): tensor of posterior distributions of nodes with shape `(#nodes,num_latent_states)`
        edge_index (torch.Tensor): tensor of edge indices with shape `(2,#edges)` that adheres to PyG specifications
        batch (torch.Tensor): vector that assigns each node to a graph id in the batch
        use_continuous_states (bool): whether or not to use the most probable state (one-hot vector) or a "soft" version
    Returns:
        a tensor of bigrams with shape `(#nodes,num_latent_states*num_latent_states)`
    """
    C = posteriors.shape[1]
    device = posteriors.get_device()
    device = 'cpu' if device == -1 else device

    if use_continuous_states:
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors.float()).repeat(1, C)
        tmp2 = posteriors.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)
    else:
        # Convert into one hot
        posteriors_one_hot = make_one_hot(posteriors.argmax(dim=1), C).float()

        # Code provided by Daniele Atzeni to speed up the computation!
        nodes_in_batch = len(batch)
        sparse_adj_matrix = torch.sparse.FloatTensor(edge_index,
                                                     torch.ones(edge_index.shape[1]).to(device),
                                                     torch.Size([nodes_in_batch, nodes_in_batch]))
        tmp1 = torch.sparse.mm(sparse_adj_matrix, posteriors_one_hot).repeat(1, C)
        tmp2 = posteriors_one_hot.reshape(-1, 1).repeat(1, C).reshape(-1, C * C)
        node_bigram_batch = torch.mul(tmp1, tmp2)

    return node_bigram_batch.float()


def make_one_hot(labels: torch.Tensor, num_unique_ids: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a vector of ids into a one-hot matrix
    Args:
        labels (torch.Tensor): the vector of ids
        num_unique_ids (torch.Tensor): number of unique ids
    Returns:
        a one-hot tensor with shape `(samples,num_unique_ids)`
    """
    device = labels.get_device()
    device = 'cpu' if device == -1 else device
    one_hot = torch.zeros(labels.size(0), num_unique_ids).to(device)
    one_hot[torch.arange(labels.size(0)).to(device), labels] = 1
    return one_hot