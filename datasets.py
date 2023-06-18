"""
  Hidden Markov Model for Temporal Graphs

  File:     datasets.py
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
from torch_geometric_temporal.dataset import (
    TwitterTennisDatasetLoader, PedalMeDatasetLoader,
    WikiMathsDatasetLoader, ChickenpoxDatasetLoader
)
from pydgn.data.dataset import TemporalDatasetInterface
import numpy as np
from os.path import join, isfile
import torch


class ChickenpoxDatasetInterface(TemporalDatasetInterface):
    """
    Chickenpox dataset.
    It contains spatio-temporal graph.
    """

    def __init__(self, root, name, lags=4, **kwargs):
        self.lags = lags

        super().__init__(root, name, **kwargs)
        self.dataset = torch.load(self.processed_file_names[0])

    def download(self):
        self.dataset = ChickenpoxDatasetLoader().get_dataset(lags=self.lags)

        path = self.raw_file_names[0]
        torch.save(self.dataset, path)

    def process(self):
        path = self.raw_file_names[0]
        raw_data = torch.load(path)

        p_path = self.processed_file_names[0]
        torch.save(raw_data, p_path)

    @property
    def raw_paths(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_dir(self):
        path = join(self.root, self.name, 'processed')
        return path

    @property
    def raw_dir(self):
        path = join(self.root, self.name, 'raw')
        return path

    @property
    def raw_file_names(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_file_names(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def processed_paths(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data


class PedalMeDatasetInterface(TemporalDatasetInterface):
    """
    PedalMe dataset.
    It contains a spatio-temporal graph for traffic forecasting.
    """

    def __init__(self, root, name='pedalme', lags=4, **kwargs):
        self.root = root
        self.name = name
        self.lags = lags
    
        super().__init__(root, name, **kwargs)
        self.dataset = torch.load(self.processed_file_names[0])

    def download(self):
        self.dataset = PedalMeDatasetLoader().get_dataset(lags=self.lags)

        path = self.raw_file_names[0]
        torch.save(self.dataset, path)

    def process(self):
        path = self.raw_file_names[0]
        raw_data = torch.load(path)

        p_path = self.processed_file_names[0]
        torch.save(raw_data, p_path)

    @property
    def raw_paths(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_dir(self):
        path = join(self.root, self.name, 'processed')
        return path

    @property
    def raw_dir(self):
        path = join(self.root, self.name, 'raw')
        return path

    @property
    def raw_file_names(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_file_names(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def processed_paths(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]
        
    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data


class WikiMathsDatasetInterface(TemporalDatasetInterface):
    """
    WikiMaths dataset.
    It contains a spatio-temporal graph for inflow passenger forecasting.
    """

    def __init__(self, root, name='wikimaths', lags=8, **kwargs):
        self.root = root
        self.name = name
        self.lags = lags
    
        super().__init__(root, name, **kwargs)
        self.dataset = torch.load(self.processed_file_names[0])
    
    def download(self):
        self.dataset = WikiMathsDatasetLoader().get_dataset(lags=self.lags)

        path = self.raw_file_names[0]
        torch.save(self.dataset, path)

    def process(self):
        path = self.raw_file_names[0]
        raw_data = torch.load(path)

        p_path = self.processed_file_names[0]
        torch.save(raw_data, p_path)

    @property
    def raw_paths(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_dir(self):
        path = join(self.root, self.name, 'processed')
        return path

    @property
    def raw_dir(self):
        path = join(self.root, self.name, 'raw')
        return path

    @property
    def raw_file_names(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_file_names(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def processed_paths(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 0

    @property
    def dim_target(self):
        return 1

    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data


# **** NODE PREDICTION ON DISCRETE-DYNAMIC GRAPHS ****
class TwitterTennisDatasetInterface(TemporalDatasetInterface):
    """
    Twitter Tennis Dataset.
    It contains Twitter mention graph related to major tennis tournaments from 2017.
    Each snapshot change with respect to edges and features.
    """

    def __init__(self, root, name, event_id='rg17', num_nodes=1000,
                 feature_mode='encoded', target_offset=1, **kwargs):
        assert event_id in ['rg17', 'uo17'], f'event_id can be rg17 or uo17, not {event_id}'
        assert num_nodes <= 1000, f'num_nodes must be less or equal to 1000, not {num_nodes}'
        assert feature_mode in [None, 'diagonal', 'encoded'], f'feature_mode can be None, diagonal, or encoded. It can not be {feature_mode}'

        self.root = root
        self.name = name
        self.event_id = event_id
        self.num_nodes = num_nodes
        self.feature_mode = feature_mode
        self.target_offset = target_offset
    
        super().__init__(root, name, **kwargs)
        self.dataset = torch.load(self.processed_file_names[0])
    
    @property
    def raw_file_names(self):
        path = join(self.root, self.name) + '.pt'
        return [path]
    
    def download(self):
        path = self.raw_file_names[0]
        self.dataset = TwitterTennisDatasetLoader(
                        event_id = self.event_id,
                        N = self.num_nodes,
                        feature_mode = self.feature_mode,
                        target_offset = self.target_offset
                    ).get_dataset()
        torch.save(self.dataset, path)
    
    def process(self):
        path = self.raw_file_names[0]
        raw_data = torch.load(path)

        p_path = self.processed_file_names[0]
        torch.save(raw_data, p_path)

    @property
    def raw_paths(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_dir(self):
        path = join(self.root, self.name, 'processed')
        return path

    @property
    def raw_dir(self):
        path = join(self.root, self.name, 'raw')
        return path

    @property
    def raw_file_names(self):
        path = join(self.root, self.name, 'raw', 'data.pt')
        return [path]

    @property
    def processed_file_names(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def processed_paths(self):
        path = join(self.root, self.name, 'processed', 'data.pt')
        return [path]

    @property
    def dim_node_features(self):
        return self.dataset.features[0].shape[1]

    @property
    def dim_edge_features(self):
        return 1

    @property
    def dim_target(self):
        # node regression: each time step is a tuple
        return 1
    
    def get_mask(self, data):
        # in this case data is a Data object containing a snapshot of a single
        # graph sequence.
        # the task is node classification at each time step
        mask = np.ones((1,1))  #  time_steps x 1
        return mask

    def __len__(self):
        return len(self.dataset.features)

    def __getitem__(self, time_index):
        data = self.dataset.__getitem__(time_index)
        setattr(data, 'mask', self.get_mask(data))
        return data