import numpy as np

from knn import KNN


class QEC(object):
    def __init__(self, knn, state_dimension, projection_type,
                 observation_dimension, buffer_size, num_actions, rng):
        self.knn = knn
        self.ec_buffer = []
        self.buffer_maximum_size = buffer_size
        self.rng = rng
        for i in range(num_actions):
            self.ec_buffer.append(KNN(buffer_size, state_dimension))

        # I tried to make a function self.projection(state)
        # but cPickle can not pickle an object that has an function attribute
        self._initialize_projection_function(state_dimension,
                                             observation_dimension,
                                             projection_type)

    def _initialize_projection_function(self, dimension_result,
                                        dimension_observation, p_type):
        if p_type == 'random':
            self.matrix_projection = self.rng.randn(dimension_result,
                                                    dimension_observation) \
                .astype(np.float32)
        elif p_type == 'VAE':
            pass
        else:
            raise ValueError('unrecognized projection type')

    def estimate(self, s, a):
        """Determine Q(s,a).

        First search in the QEC-table if an entry already exists (O(N)).
        Otherwise estimate by KNN (O(N*D*logK)).
        """
        state = np.dot(self.matrix_projection, s.flatten())
        action_buffer = self.ec_buffer[a]
        q_value = action_buffer.peek(state, None, modify=False)
        if q_value is not None:
            return q_value
        # If the number of elements in the action-buffer is smaller than k
        # then return an 'infinitely' high reward to make sure that this action
        # gets explored, otherwise an estimation by the knn-algorithm
        # can not be performed and throws an exception,
        # because it needs at least k elements as neighbors.
        # TODO: Consider implementing as -inf or filling up action-buffers
        # randomly before anything else
        elif action_buffer.curr_capacity < self.knn:
            return float('inf')

        return action_buffer.knn_value(state, self.knn)

    def update(self, s, a, r):
        """s is 84*84*3;  a is 0 to num_actions; r is reward"""
        state = np.dot(self.matrix_projection, s.flatten())
        q_value = self.ec_buffer[a].peek(state, r, modify=True)
        if q_value is None:
            self.ec_buffer[a].add(state, r)
