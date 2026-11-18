
import numpy as np


class ViterbiDecoder:
    """Decode the most likely sequence of road candidates."""

    def decode(
        self,
        observations,
        candidates_by_observation,
        emissions_by_observation,
        transition_scorer,
    ):
        """
        Decode candidate sequences independently across continuous
        segments of observations.

        Returns:
            List of Candidate objects or None, preserving the
            original GPS observation order.
        """

        if not observations:
            return []

        states = []
        emission_scores = []

        # Step 1: Prepare states and emission scores.
        for observation in observations:
            index = observation.observation_index
            candidates = candidates_by_observation.get(index, [])

            if not candidates:
                states.append([])
                emission_scores.append(
                    np.array([], dtype=float)
                )
                continue

            if index not in emissions_by_observation:
                raise ValueError(
                    f"Missing emission scores at observation {index}"
                )

            scores = np.asarray(
                emissions_by_observation[index],
                dtype=float,
            )

            if len(scores) != len(candidates):
                raise ValueError(
                    f"Emission count mismatch at observation {index}"
                )

            if not np.all(np.isfinite(scores)):
                raise ValueError(
                    f"Non-finite emission scores at observation {index}"
                )

            states.append(candidates)
            emission_scores.append(scores)

        # Preserve missing observations as None.
        result = [None] * len(observations)

        # Step 2: Find and decode each continuous segment.
        segment_start = 0
        total = len(observations)

        while segment_start < total:

            # Skip observations without candidates.
            if not states[segment_start]:
                segment_start += 1
                continue

            # Find the end of this continuous segment.
            segment_end = segment_start

            while (
                segment_end + 1 < total
                and states[segment_end + 1]
            ):
                segment_end += 1

            # Step 3: Initialize dynamic programming.
            dp = [emission_scores[segment_start].copy()]
            backpointers = [
                np.full(
                    len(states[segment_start]),
                    -1,
                    dtype=int,
                )
            ]

            # Step 4: Forward pass.
            for t in range(segment_start + 1, segment_end + 1):

                previous_states = states[t - 1]
                current_states = states[t]

                current_dp = np.full(
                    len(current_states),
                    -np.inf,
                    dtype=float,
                )

                current_back = np.full(
                    len(current_states),
                    -1,
                    dtype=int,
                )

                previous_observation = observations[t - 1]
                current_observation = observations[t]

                for j, current_candidate in enumerate(current_states):

                    best_score = -np.inf
                    best_previous = -1

                    for i, previous_candidate in enumerate(
                        previous_states
                    ):

                        transition = transition_scorer.score(
                            previous_observation,
                            previous_candidate,
                            current_observation,
                            current_candidate,
                        )

                        score = dp[-1][i] + transition

                        if score > best_score:
                            best_score = score
                            best_previous = i

                    current_dp[j] = (
                        best_score + emission_scores[t][j]
                    )

                    current_back[j] = best_previous

                dp.append(current_dp)
                backpointers.append(current_back)

            # Step 5: Backtracking.
            best_index = int(np.argmax(dp[-1]))

            for local_t in range(
                segment_end - segment_start,
                -1,
                -1,
            ):

                original_t = segment_start + local_t

                result[original_t] = states[original_t][best_index]

                if local_t > 0:
                    best_index = int(
                        backpointers[local_t][best_index]
                    )

                    if best_index < 0:
                        raise RuntimeError(
                            f"Invalid backpointer at observation "
                            f"{original_t}"
                        )

            # Move to the next segment.
            segment_start = segment_end + 1

        return result
