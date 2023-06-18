"""
  Hidden Markov Model for Temporal Graphs

  File:     hmm4g_embedding_task.py
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
import shutil

import torch


# This works with graph classification only
from pydgn.experiment.experiment import Experiment
from pydgn.static import LOSS, SCORE
from torch_geometric.data import Data


class EmbeddingHMM4GTask(Experiment):
    def __init__(self, model_configuration, exp_path, exp_seed):
        super().__init__(model_configuration, exp_path, exp_seed)
        self.root_exp_path = exp_path  # to distinguish from layers' exp_path
        self.output_folder = os.path.join(exp_path, 'outputs')

    def _create_unsup_embeddings(self, mode, max_layers):
        # Load previous outputs if any according to prev. layers to consider (ALL TENSORS)
        embeddings_list = []
        for layer_id in range(1, max_layers+1):
            node_embeddings_list, _ = self._load_outputs(mode, layer_id)

            for time_index in range(len(node_embeddings_list)):
                ne = node_embeddings_list[time_index].unsqueeze(1)
                if layer_id == 1:
                    embeddings_list.append(ne)
                else:
                    # concat over layers
                    embeddings_list[time_index] = torch.cat((embeddings_list[time_index], ne), dim=1)

        # create a new dimension for time steps
        embeddings_list = torch.stack(embeddings_list, dim=1)
        return embeddings_list

    def _create_extra_dataset(self, prev_layer_id, mode):
        if prev_layer_id == 0:
            return None

        node_embeddings_list, stats_list = self._load_outputs(mode, prev_layer_id)

        data_list = []

        assert len(node_embeddings_list) == len(stats_list)

        no_timesteps = len(node_embeddings_list)
        for time_step in range(no_timesteps):
            data_list.append(Data(embeddings=node_embeddings_list[time_step], stats=stats_list[time_step]))

        return data_list

    def _load_outputs(self, mode, prev_layer_id):
        # Load previous layer

        embeddings_list = torch.load(os.path.join(self.output_folder, mode, f'embeddings_{prev_layer_id}.pt'))
        stats_list = torch.load(os.path.join(self.output_folder, mode, f'stats_{prev_layer_id}.pt'))

        return embeddings_list, stats_list

    def _store_outputs(self, mode, depth, embeddings, stats):

        if not os.path.exists(os.path.join(self.output_folder, mode)):
            os.makedirs(os.path.join(self.output_folder, mode))

        embeddings_filepath = os.path.join(self.output_folder, mode, f'embeddings_{depth}.pt')
        torch.save([emb_tensor for emb_tensor in embeddings], embeddings_filepath)

        stats_filepath = os.path.join(self.output_folder, mode, f'stats_{depth}.pt')
        torch.save([stats_tensor for stats_tensor in stats], stats_filepath)

    def run_valid(self, dataset_getter, logger):
        """
        This function returns the training and validation or test accuracy
        :return: (training accuracy, validation/test accuracy)
        """

        batch_size = self.model_config.layer_config['batch_size']
        shuffle = self.model_config.layer_config['shuffle'] \
            if 'shuffle' in self.model_config.layer_config else True

        assert shuffle == False, "shuffle cannot be true in single-graph sequence experiments"

        depth = 1
        max_layers = self.model_config.layer_config['max_layers']
        while depth <= max_layers:

            # Change exp path to allow Stop & Resume
            self.exp_path = os.path.join(self.root_exp_path, f'layer_{depth}')
            if os.path.exists(os.path.join(self.root_exp_path, f'layer_{depth + 1}')):
                # print("skip layer", depth)
                depth += 1
                continue

            train_out = self._create_extra_dataset(mode='train', prev_layer_id=depth-1)
            val_out = self._create_extra_dataset(mode='validation', prev_layer_id=depth-1)
            train_loader = dataset_getter.get_inner_train(batch_size=batch_size, **dict(shuffle=False, extra=train_out))
            val_loader = dataset_getter.get_inner_val(batch_size=batch_size, **dict(shuffle=False, extra=val_out))

            # Instantiate the Dataset
            dim_node_features = dataset_getter.get_dim_node_features()
            dim_edge_features = dataset_getter.get_dim_edge_features()
            dim_target = dataset_getter.get_dim_target()

            # ==== # WARNING: WE ARE JUST PRECOMPUTING OUTER_TEST EMBEDDINGS FOR SUBSEQUENT CLASSIFIERS
            # WE ARE NOT TRAINING ON TEST (EVEN THOUGH UNSUPERVISED)
            # ==== #

            test_out = self._create_extra_dataset(mode='test', prev_layer_id=depth-1)
            test_loader = dataset_getter.get_outer_test(batch_size=batch_size, **dict(shuffle=False, extra=test_out))

            # ==== #

            # Instantiate the Model
            new_layer = self.create_incremental_model(dim_node_features, dim_edge_features, dim_target, depth,
                                                      prev_outputs_to_consider=[depth-1])

            # Instantiate the engine (it handles the training loop and the inference phase by abstracting the specifics)
            incremental_training_engine = self.create_incremental_engine(new_layer)

            train_loss, train_score, train_out, \
            val_loss, val_score, val_out, \
            _, _, test_out = incremental_training_engine.train(train_loader=train_loader,
                                                                validation_loader=val_loader,
                                                                test_loader=test_loader,
                                                                max_epochs=self.model_config.layer_config['epochs'],
                                                                logger=logger)

            for loader, out, mode in [(train_loader, train_out, 'train'), (val_loader, val_out, 'validation'),
                                      (test_loader, test_out, 'test')]:
                # Store outputs
                embeddings, stats = out
                self._store_outputs(mode, depth, embeddings, stats)

            depth += 1

        # NOW LOAD ALL EMBEDDINGS AND STORE THE EMBEDDINGS DATASET ON a torch file.

        train_out_emb = self._create_unsup_embeddings(mode='train', max_layers=max_layers)
        val_out_emb = self._create_unsup_embeddings(mode='validation', max_layers=max_layers)
        test_out_emb = self._create_unsup_embeddings(mode='test', max_layers=max_layers)

        # Necessary info to give a unique name to the dataset (some hyper-params like epochs are assumed to be fixed)
        embeddings_folder = self.model_config.layer_config['embeddings_folder']
        max_layers = self.model_config.layer_config['max_layers']
        unibigram = self.model_config.layer_config['unibigram']
        C = self.model_config.layer_config['C']
        outer_k = dataset_getter.outer_k
        inner_k = dataset_getter.inner_k
        # ====

        if not os.path.exists(os.path.join(embeddings_folder, dataset_getter.dataset_name)):
            os.makedirs(os.path.join(embeddings_folder, dataset_getter.dataset_name))

        # Alpha has dimension C
        unigram_dim = C
        assert unibigram == True

        # Retrieve store both unigram and bigram separately
        for unib in [False, True]:
            base_path = os.path.join(embeddings_folder, dataset_getter.dataset_name,
                                     f'{max_layers}_{unib}_{C}_{outer_k + 1}_{inner_k + 1}')
            torch.save(train_out_emb if unib else train_out_emb[:, :, :, unigram_dim], base_path + '_train.torch')
            torch.save(val_out_emb if unib else val_out_emb[:, :, :, unigram_dim], base_path + '_val.torch')
            torch.save(test_out_emb if unib else test_out_emb[:, :, :, unigram_dim], base_path + '_test.torch')

        # CLEAR OUTPUTS
        for mode in ['train', 'validation', 'test']:
            shutil.rmtree(os.path.join(self.output_folder, mode), ignore_errors=True)

        tr_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        vl_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        return tr_res, vl_res

    def run_test(self, dataset_getter, logger):
        tr_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        vl_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        te_res = {LOSS: {'main_loss': torch.zeros(1)}, SCORE: {'main_score': torch.zeros(1)}}
        return tr_res, vl_res, te_res
