"""
  Hidden Markov Model for Temporal Graphs

  File:     hmm4g_classifier_task.py
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
import os

import torch
from pydgn.data.provider import SingleGraphSequenceDataProvider
from pydgn.data.sampler import RandomSampler
from pydgn.experiment.experiment import Experiment

from pydgn.experiment.util import s2c
from pydgn.static import LOSS, SCORE
from torch_geometric.data import Data


# This works with graph classification only
class ClassifierHMM4GTask(Experiment):
    def __init__(self, model_configuration, exp_path, exp_seed):
        super().__init__(model_configuration, exp_path, exp_seed)
        self.root_exp_path = exp_path  # to distinguish from layers' exp_path
        self.output_folder = os.path.join(exp_path, 'outputs')

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        max_layers = self.model_config.layer_config['max_layers']
        layers = self.model_config.layer_config['layers']
        unibigram = self.model_config.layer_config['unibigram']
        C = self.model_config.layer_config['C']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # ====

        base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                 f'{max_layers}_{unibigram}_{C}_{outer_k + 1}_{inner_k + 1}')
        train_out_emb = torch.load(base_path + '_train.torch')[:, :, :layers, :]
        val_out_emb = torch.load(base_path + '_val.torch')[:, :, :layers, :]
        train_out_emb = torch.reshape(train_out_emb, (train_out_emb.shape[0], train_out_emb.shape[1], -1))
        val_out_emb = torch.reshape(val_out_emb, (val_out_emb.shape[0], val_out_emb.shape[1], -1))

        # Recover the targets
        fake_train_loader = dataset_getter.get_inner_train(batch_size=1, shuffle=False)
        fake_val_loader = dataset_getter.get_inner_val(batch_size=1, shuffle=False)
        train_y = [el.y for el in fake_train_loader.dataset]
        val_y = [el.y for el in fake_val_loader.dataset]
        try: 
            train_link_pred_ids = [el.link_pred_ids for el in fake_train_loader.dataset]
            val_link_pred_ids = [el.link_pred_ids for el in fake_val_loader.dataset]
        except: 
            train_link_pred_ids = []
            val_link_pred_ids = []

        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # build data lists
        if len(train_link_pred_ids) == 0:
            train_list = [Data(x=train_out_emb[:, time_index], y=train_y[time_index]) for time_index in range(train_out_emb.shape[1])]
            val_list = [Data(x=val_out_emb[:, time_index], y=val_y[time_index]) for time_index in range(val_out_emb.shape[1])]
        else:
            train_list = [Data(x=train_out_emb[:, time_index], y=train_y[time_index], link_pred_ids=train_link_pred_ids[time_index]) for time_index in range(train_out_emb.shape[1])]
            val_list = [Data(x=val_out_emb[:, time_index], y=val_y[time_index], link_pred_ids=val_link_pred_ids[time_index]) for time_index in range(val_out_emb.shape[1])]

        if not arbitrary_logic_batch_size:
            train_loader = torch.utils.data.DataLoader(train_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle,
                                      collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            val_loader = torch.utils.data.DataLoader(val_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle,
                                    collate_fn=SingleGraphSequenceDataProvider.collate_fn)
        else:
            train_loader = torch.utils.data.DataLoader(train_list, sampler=RandomSampler(train_list), batch_size=arbitrary_logic_batch_size,
                                      collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            val_loader = torch.utils.data.DataLoader(val_list, sampler=RandomSampler(val_list), batch_size=arbitrary_logic_batch_size,
                                    collate_fn=SingleGraphSequenceDataProvider.collate_fn)

        # Instantiate the Dataset
        dim_features = train_out_emb.shape[2]
        dim_target = dataset_getter.get_dim_target()

        config = self.model_config.layer_config['arbitrary_function_config']
        device = config['device']

        predictor_class = s2c(config['readout'])
        model = predictor_class(dim_node_features=dim_features,
                                dim_edge_features=0,
                                dim_target=dim_target,
                                config=config)

        reset_eval_model_hidden_state = self.model_config['reset_eval_model_hidden_state']
        predictor_engine = self._create_engine(config, model, device, evaluate_every=self.model_config.evaluate_every,
                                               reset_eval_model_hidden_state=reset_eval_model_hidden_state)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        _, _, _ = predictor_engine.train(train_loader=train_loader,
                                          validation_loader=val_loader,
                                          test_loader=None,
                                          max_epochs=config['epochs'],
                                          logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        return train_res, val_res

    def run_test(self, dataset_getter, logger):
        """
        This function returns the training and test accuracy. DO NOT USE THE TEST FOR ANY REASON
        :return: (training accuracy, test accuracy)
        """

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        max_layers = self.model_config.layer_config['max_layers']
        layers = self.model_config.layer_config['layers']
        unibigram = self.model_config.layer_config['unibigram']
        C = self.model_config.layer_config['C']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        if inner_k is None:  # workaround the "safety" procedure of evaluation protocol, but we will not do anything wrong.
            dataset_getter.set_inner_k(0)
            inner_k = 0  # pick the split of the first inner fold
        # ====

        # NOTE: We reload the associated inner train and val splits, using the outer_test for assessment.
        # This is slightly different from standard exps, where we compute a different outer train-val split, but it should not change things much.

        base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                 f'{max_layers}_{unibigram}_{C}_{outer_k + 1}_{inner_k + 1}')
        train_out_emb = torch.load(base_path + '_train.torch')[:, :, :layers, :]
        val_out_emb = torch.load(base_path + '_val.torch')[:, :, :layers, :]
        test_out_emb = torch.load(base_path + '_test.torch')[:, :, :layers, :]
        train_out_emb = torch.reshape(train_out_emb, (train_out_emb.shape[0], train_out_emb.shape[1], -1))
        val_out_emb = torch.reshape(val_out_emb, (val_out_emb.shape[0], val_out_emb.shape[1], -1))
        test_out_emb = torch.reshape(test_out_emb, (test_out_emb.shape[0], test_out_emb.shape[1], -1))

        # Recover the targets
        fake_train_loader = dataset_getter.get_inner_train(batch_size=1, shuffle=False)
        fake_val_loader = dataset_getter.get_inner_val(batch_size=1, shuffle=False)
        fake_test_loader = dataset_getter.get_outer_test(batch_size=1, shuffle=False)
        train_y = [el.y for el in fake_train_loader.dataset]
        val_y = [el.y for el in fake_val_loader.dataset]
        test_y = [el.y for el in fake_test_loader.dataset]
        try: 
            train_link_pred_ids = [el.link_pred_ids for el in fake_train_loader.dataset]
            val_link_pred_ids = [el.link_pred_ids for el in fake_val_loader.dataset]
            test_link_pred_ids = [el.link_pred_ids for el in fake_test_loader.dataset]
        except: 
            train_link_pred_ids = []
            val_link_pred_ids = []
            test_link_pred_ids = []

        arbitrary_logic_batch_size = self.model_config.layer_config['arbitrary_function_config']['batch_size']
        arbitrary_logic_shuffle = self.model_config.layer_config['arbitrary_function_config']['shuffle'] \
            if 'shuffle' in self.model_config.layer_config['arbitrary_function_config'] else True

        # build data lists
        if len(train_link_pred_ids) == 0:
            train_list = [Data(x=train_out_emb[:, time_index], y=train_y[time_index]) for time_index in range(train_out_emb.shape[1])]
            val_list = [Data(x=val_out_emb[:, time_index], y=val_y[time_index]) for time_index in range(val_out_emb.shape[1])]
            test_list = [Data(x=test_out_emb[:, time_index], y=test_y[time_index]) for time_index in range(test_out_emb.shape[1])]
        else:
            train_list = [Data(x=train_out_emb[:, time_index], y=train_y[time_index], link_pred_ids=train_link_pred_ids[time_index]) for time_index in range(train_out_emb.shape[1])]
            val_list = [Data(x=val_out_emb[:, time_index], y=val_y[time_index], link_pred_ids=val_link_pred_ids[time_index]) for time_index in range(val_out_emb.shape[1])]
            test_list = [Data(x=test_out_emb[:, time_index], y=test_y[time_index], link_pred_ids=test_link_pred_ids[time_index]) for time_index in range(test_out_emb.shape[1])]


        if not arbitrary_logic_batch_size:
            train_loader = torch.utils.data.DataLoader(train_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle,
                                      collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            val_loader = torch.utils.data.DataLoader(val_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle,
                                    collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            test_loader = torch.utils.data.DataLoader(test_list, batch_size=arbitrary_logic_batch_size, shuffle=arbitrary_logic_shuffle,
                                     collate_fn=SingleGraphSequenceDataProvider.collate_fn)
        else:
            train_loader = torch.utils.data.DataLoader(train_list, sampler=RandomSampler(train_list), batch_size=arbitrary_logic_batch_size,
                                      collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            val_loader = torch.utils.data.DataLoader(val_list, sampler=RandomSampler(val_list), batch_size=arbitrary_logic_batch_size,
                                    collate_fn=SingleGraphSequenceDataProvider.collate_fn)
            test_loader = torch.utils.data.DataLoader(test_list, sampler=RandomSampler(test_list), batch_size=arbitrary_logic_batch_size,
                                     collate_fn=SingleGraphSequenceDataProvider.collate_fn)

        # Instantiate the Dataset
        dim_features = train_out_emb.shape[2]
        dim_target = dataset_getter.get_dim_target()

        config = self.model_config.layer_config['arbitrary_function_config']
        device = config['device']

        predictor_class = s2c(config['readout'])
        model = predictor_class(dim_node_features=dim_features,
                                dim_edge_features=0,
                                dim_target=dim_target,
                                config=config)

        reset_eval_model_hidden_state = self.model_config['reset_eval_model_hidden_state']
        predictor_engine = self._create_engine(config, model, device, evaluate_every=self.model_config.evaluate_every,
                                               reset_eval_model_hidden_state=reset_eval_model_hidden_state)

        train_loss, train_score, _, \
        val_loss, val_score, _, \
        test_loss, test_score, _ = predictor_engine.train(train_loader=train_loader,
                                                           validation_loader=val_loader,
                                                           test_loader=test_loader,
                                                           max_epochs=config['epochs'],
                                                           logger=logger)

        train_res = {LOSS: train_loss, SCORE: train_score}
        val_res = {LOSS: val_loss, SCORE: val_score}
        test_res = {LOSS: test_loss, SCORE: test_score}
        return train_res, val_res, test_res
