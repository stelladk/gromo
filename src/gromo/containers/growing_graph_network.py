import copy
import operator
from typing import Iterator, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_dag import GrowingDAG
from gromo.utils.utils import f1_micro


class GrowingGraphNetwork(GrowingContainer):
    """Growing DAG Network

    Parameters
    ----------
    in_features : int, optional
        size of input features, by default 5
    out_features : int, optional
        size of output dimension, by default 1
    use_bias : bool, optional
        automatically use bias in the layers, by default True
    use_batch_norm : bool, optional
        use batch normalization on the last layer, by default False
    neurons : int, optional
        default number of neurons to add at each step, by default 20
    device : str | None, optional
        default device, by default None
    """

    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 1,
        neurons: int = 20,
        use_bias: bool = True,
        use_batch_norm: bool = False,
        layer_type: str = "linear",
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        super(GrowingGraphNetwork, self).__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            layer_type=layer_type,
            seed=seed,
            device=device,
        )
        self.use_batch_norm = use_batch_norm
        self.neurons = neurons

        self.global_step = 0
        self.global_epoch = 0
        self.loss_fn = nn.CrossEntropyLoss()

        self.reset_network()

    def init_empty_graph(self) -> None:
        """Create empty DAG with start and end nodes"""
        self.dag = GrowingDAG(
            in_features=self.in_features,
            out_features=self.out_features,
            neurons=self.neurons,
            use_bias=self.use_bias,
            use_batch_norm=self.use_batch_norm,
            layer_type=self.layer_type,
            device=self.device,
        )

        if (self.dag.root, self.dag.end) in self.dag.edges:
            self.dag.remove_edge(self.dag.root, self.dag.end)

    def reset_network(self) -> None:
        """Reset graph to empty"""
        self.init_empty_graph()
        self.global_step = 0
        self.global_epoch = 0
        self.growth_history = {}
        self.growth_history_step()

    def growth_history_step(
        self, neurons_added: list = [], neurons_updated: list = [], nodes_added: list = []
    ) -> None:
        """Record recent modifications on history dictionary

        Parameters
        ----------
        neurons_added : list, optional
            list of edges that were added or increased in dimension, by default []
        neurons_updated : list, optional
            list of edges whose weights were updated, by default []
        nodes_added : list, optional
            list of nodes that were added, by default []
        """
        # TODO: keep track of updated edges/neurons_updated
        if self.global_step not in self.growth_history:
            self.growth_history[self.global_step] = {}

        keep_max = lambda new_value, key: max(
            self.growth_history[self.global_step].get(key, 0), new_value
        )

        step = {}
        for edge in self.dag.edges:
            new_value = (
                2 if edge in neurons_added else 1 if edge in neurons_updated else 0
            )
            step[str(edge)] = keep_max(new_value, str(edge))

        for node in self.dag.nodes:
            new_value = 2 if node in nodes_added else 0
            step[str(node)] = keep_max(new_value, str(node))

        self.growth_history[self.global_step].update(step)

    def execute_expansions(
        self,
        generations: list[dict],
        bottleneck: dict,
        input_B: dict,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_dev: torch.Tensor,
        Y_dev: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        amplitude_factor: bool,
        verbose: bool = False,
    ) -> None:
        """Execute all DAG expansions and save statistics

        Parameters
        ----------
        generations : list[dict]
            list of dictionaries with growth actions information
        bottleneck : dict
            dictionary of calculated expressivity bottleneck at each pre-activity
        input_B : dict
            dictionary of post-activity input of each node
        X_train : torch.Tensor
            train features
        Y_train : torch.Tensor
            train labels
        X_dev : torch.Tensor
            development features
        Y_dev : torch.Tensor
            development labels
        X_val : torch.Tensor
            validation features
        Y_val : torch.Tensor
            validation labels
        amplitude_factor : bool
            use amplitude factor on new neurons
        verbose : bool, optional
            print info, by default False
        """
        # Execute all graph growth options
        for gen in generations:
            # Create a new edge
            if gen.get("type") == "edge":
                attributes = gen.get("attributes", {})
                prev_node = attributes.get("previous_node")
                next_node = attributes.get("next_node")

                if verbose:
                    print(f"Adding direct edge from {prev_node} to {next_node}")

                model_copy = copy.deepcopy(self)
                model_copy.to(self.device)
                model_copy.dag.add_direct_edge(
                    prev_node, next_node, attributes.get("edge_attributes", {})
                )

                model_copy.growth_history_step(neurons_added=[(prev_node, next_node)])

                # Update weight of next_node's incoming edge
                model_copy.dag.update_edge_weights(
                    prev_node=prev_node,
                    next_node=next_node,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    x=X_dev,
                    y=Y_dev,
                    amplitude_factor=amplitude_factor,
                    verbose=verbose,
                )

                # TODO: save updates weight tensors
                # gen[] =

            # Create/Expand node
            elif gen.get("type") == "node":
                attributes = gen.get("attributes", {})
                new_node = attributes.get("new_node")
                prev_nodes = attributes.get("previous_node")
                next_nodes = attributes.get("next_node")
                new_edges = attributes.get("new_edges")

                # copy.deepcopy(self.dag)
                model_copy = copy.deepcopy(self)
                model_copy.to(self.device)

                if new_node not in model_copy.dag.nodes:
                    model_copy.dag.add_node_with_two_edges(
                        prev_nodes,
                        new_node,
                        next_nodes,
                        attributes.get("node_attributes"),
                        attributes.get("edge_attributes", {}),
                    )
                    prev_nodes = [prev_nodes]
                    next_nodes = [next_nodes]

                model_copy.growth_history_step(
                    nodes_added=new_node, neurons_added=new_edges
                )

                # Update weights of new edges
                model_copy.dag.expand_node(
                    node=new_node,
                    prev_nodes=prev_nodes,
                    next_nodes=next_nodes,
                    bottlenecks=bottleneck,
                    activities=input_B,
                    x=X_dev,
                    y=Y_dev,
                    amplitude_factor=amplitude_factor,
                    verbose=verbose,
                )

                # TODO: save update weight tensors
                # gen[] =

            # Evaluate
            acc_train, loss_train = self.evaluate(X_train, Y_train)
            acc_dev, loss_dev = self.evaluate(X_dev, Y_dev)
            acc_val, loss_val = model_copy.evaluate(X_val, Y_val)

            # TODO: return all info instead of saving
            gen["loss_train"] = loss_train
            gen["loss_dev"] = loss_dev
            gen["loss_val"] = loss_val
            gen["acc_train"] = acc_train
            gen["acc_dev"] = acc_dev
            gen["acc_val"] = acc_val
            gen["nb_params"] = model_copy.dag.count_parameters_all()
            gen["BIC"] = model_copy.BIC(loss_val, n=len(X_val))

            # TEMP: save DAG
            gen["dag"] = model_copy.dag
            gen["growth_history"] = model_copy.growth_history

        del model_copy

    def restrict_action_space(
        self, generations: list[dict], chosen_position: str
    ) -> list[dict]:
        """Reduce action space to contribute only to specific node position

        Parameters
        ----------
        generations : list[dict]
            list of dictionaries with growth actions information
        chosen_position : str
            node position to restrict to

        Returns
        -------
        list[dict]
            reduced list of dictionaries with growth actions information
        """
        new_generations = []
        for gen in generations:
            new_node = gen["attributes"].get("new_node", -1)
            next_node = gen["attributes"].get("next_node", -1)
            if new_node == chosen_position:
                # Case: expand current node
                new_generations.append(gen)
            if isinstance(next_node, list) and chosen_position in next_node:
                # Case: expand immediate previous node
                new_generations.append(gen)
            elif next_node == chosen_position:
                # Case: add new previous node
                new_generations.append(gen)
        return new_generations

    def choose_growth_best_action(
        self, options: list[dict], use_bic: bool = False, verbose: bool = False
    ) -> None:
        """Choose the growth action with the minimum validation loss greedily
        Log average metrics of the current growth step
        Reconstruct chosen graph and discard the rest

        Parameters
        ----------
        options : list[dict]
            dictionary with all possible graphs and their statistics
        use_bic : bool, optional
            use BIC to select the network expansion, by default False
        verbose : bool, optional
            print info, by default False
        """
        # Greedy choice based on validation loss
        selection = {}
        if use_bic:
            for index, item in enumerate(options):
                selection[index] = item["BIC"]
        else:
            for index, item in enumerate(options):
                selection[index] = item["loss_val"]

        best_ind = min(selection.items(), key=operator.itemgetter(1))[0]

        if verbose:
            print("Chose option", best_ind)

        # Reconstruct graph
        best_option = options[best_ind]
        del options

        self.dag = copy.copy(best_option["dag"])
        self.growth_history = best_option["growth_history"]
        self.growth_loss_train = best_option["loss_train"]
        self.growth_loss_dev = best_option["loss_dev"]
        self.growth_loss_val = best_option["loss_val"]
        self.growth_acc_train = best_option["acc_train"]
        self.growth_acc_dev = best_option["acc_dev"]
        self.growth_acc_val = best_option["acc_val"]
        del best_option

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DAG network

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output of model
        """
        return self.dag(x)

    def extended_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of DAG network including extensions of the modules

        Parameters
        ----------
        x : torch.Tensor
            input tensor

        Returns
        -------
        torch.Tensor
            output of the extended model
        """
        return self.dag.extended_forward(x)

    def parameters(self) -> Iterator:
        """Iterator of network parameters

        Yields
        ------
        Iterator
            parameters iterator
        """
        return self.dag.parameters()

    def BIC(self, loss: float, n: int) -> float:
        """Bayesian Information Criterion
        BIC = k*log(n) - 2log(L), where k is the number of parameters

        Parameters
        ----------
        loss : float
            loss of the model
        n : int
            number of samples used for training

        Returns
        -------
        float
            BIC score
        """
        k = self.dag.count_parameters_all()
        return k * np.log2(n) - 2 * np.log2(loss)

    # TODO: can we remove this?
    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        with_f1score: bool = False,
    ) -> Union[tuple[float, float], tuple[float, float, float]]:
        """Evaluate network on batch

        Important: Assumes that the batch is already on the correct device

        Parameters
        ----------
        x : torch.Tensor
            input features tensor
        y : torch.Tensor
            true labels tensor
        with_f1score : bool, optional
            calculate f1-score, by default False

        Returns
        -------
        tuple[float, float] | tuple[float, float, float]
            accuracy and loss, optionally f1-score
        """
        with torch.no_grad():
            pred = self(x)
            loss = self.loss_fn(pred, y)

        if self.out_features > 1:
            final_pred = pred.argmax(axis=1)
            correct = (final_pred == y).int().sum()
            accuracy = (correct / pred.shape[0]).item()
        else:
            accuracy = -1

        if with_f1score:
            if self.out_features > 1:
                f1score = f1_micro(y.cpu(), final_pred.cpu())
            else:
                f1score = -1
            return accuracy, loss.item(), f1score

        return accuracy, loss.item()

    # TODO: can we remove this
    def evaluate_dataset(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate network on dataset

        Parameters
        ----------
        dataloader : DataLoader
            dataloader containing the data

        Returns
        -------
        tuple[float, float]
            accuracy and loss
        """
        correct, total = 0, 0

        loss = []
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                pred = self(x)
                loss.append(self.loss_fn(pred, y).item())

            final_pred = pred.argmax(axis=1)
            count_this = final_pred == y
            count_this = count_this.sum()

            correct += count_this.item()
            total += len(pred)

        return (correct / total), np.mean(loss).item()
