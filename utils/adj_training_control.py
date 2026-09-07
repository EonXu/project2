"""Pure-Python control helpers for adjacency PPO training."""

import math


def _finite_metric(train_info, key):
    if key not in train_info:
        raise RuntimeError(
            "adjacency training is missing required {} metric".format(key)
        )
    value = float(train_info[key])
    if not math.isfinite(value):
        raise RuntimeError(
            "adjacency training {} metric must be finite".format(key)
        )
    return value


def _population_ratio(numerator, denominator, label):
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise RuntimeError(
            "adjacency {} control population must be finite".format(label)
        )
    if numerator < 0.0 or denominator < 0.0:
        raise RuntimeError(
            "adjacency {} control population must be nonnegative".format(
                label
            )
        )
    tolerance = 1e-9 * max(1.0, abs(denominator))
    if numerator > denominator + tolerance:
        raise RuntimeError(
            "adjacency {} control numerator exceeds its denominator".format(
                label
            )
        )
    if denominator == 0.0:
        if numerator != 0.0:
            raise RuntimeError(
                "adjacency {} zero denominator has nonzero mass".format(
                    label
                )
            )
        return 0.0
    return numerator / denominator


def select_adj_control_population(train_info, use_stale_trust):
    """Return one mini-batch's exact graph/factor control population.

    Ratios are diagnostics.  Numerators and denominators are the composable
    state used by epoch/update control so differently sized valid populations
    cannot be given equal weight accidentally.
    """
    if use_stale_trust:
        population_name = "trusted"
        graph_ratio_key = "trusted_clamp_ratio"
        factor_ratio_key = "trusted_factor_clamp_ratio"
    else:
        population_name = "raw"
        graph_ratio_key = "clamp_ratio"
        factor_ratio_key = "factor_clamp_ratio"

    graph_numerator = _finite_metric(
        train_info,
        "_adj_control_{}_graph_numerator".format(population_name),
    )
    graph_denominator = _finite_metric(
        train_info,
        "_adj_control_{}_graph_denominator".format(population_name),
    )
    factor_numerator = _finite_metric(
        train_info,
        "_adj_control_{}_factor_numerator".format(population_name),
    )
    factor_denominator = _finite_metric(
        train_info,
        "_adj_control_{}_factor_denominator".format(population_name),
    )
    graph_ratio = _population_ratio(
        graph_numerator,
        graph_denominator,
        "{} graph".format(population_name),
    )
    factor_ratio = _population_ratio(
        factor_numerator,
        factor_denominator,
        "{} factor".format(population_name),
    )
    reported_graph_ratio = _finite_metric(train_info, graph_ratio_key)
    reported_factor_ratio = _finite_metric(train_info, factor_ratio_key)
    if (
            abs(reported_graph_ratio - graph_ratio) > 1e-6
            or abs(reported_factor_ratio - factor_ratio) > 1e-6):
        raise RuntimeError(
            "adjacency {} control ratio disagrees with its "
            "numerator/denominator".format(population_name)
        )
    return {
        "population": population_name,
        "graph_numerator": graph_numerator,
        "graph_denominator": graph_denominator,
        "graph_ratio": graph_ratio,
        "graph_valid": float(graph_denominator > 0.0),
        "factor_numerator": factor_numerator,
        "factor_denominator": factor_denominator,
        "factor_ratio": factor_ratio,
        "factor_valid": float(factor_denominator > 0.0),
    }


def aggregate_adj_control_populations(populations):
    """Aggregate exact mini-batch masses into one epoch/update population."""
    populations = list(populations)
    if not populations:
        raise RuntimeError("adjacency control population is empty")
    names = set(population["population"] for population in populations)
    if len(names) != 1:
        raise RuntimeError(
            "adjacency control aggregation mixed raw and trusted populations"
        )
    graph_numerator = sum(
        float(population["graph_numerator"])
        for population in populations
    )
    graph_denominator = sum(
        float(population["graph_denominator"])
        for population in populations
    )
    factor_numerator = sum(
        float(population["factor_numerator"])
        for population in populations
    )
    factor_denominator = sum(
        float(population["factor_denominator"])
        for population in populations
    )
    population_name = names.pop()
    return {
        "population": population_name,
        "graph_numerator": graph_numerator,
        "graph_denominator": graph_denominator,
        "graph_ratio": _population_ratio(
            graph_numerator,
            graph_denominator,
            "{} graph aggregate".format(population_name),
        ),
        "graph_valid": float(graph_denominator > 0.0),
        "factor_numerator": factor_numerator,
        "factor_denominator": factor_denominator,
        "factor_ratio": _population_ratio(
            factor_numerator,
            factor_denominator,
            "{} factor aggregate".format(population_name),
        ),
        "factor_valid": float(factor_denominator > 0.0),
    }


def select_adj_control_ratios(train_info, use_stale_trust):
    """Return graph/factor clamp ratios for PPO and replay control.

    When stale-trust weighting is active, the loss excludes or downweights
    stale transitions. Control decisions must therefore use the matching
    trusted population rather than the unweighted diagnostic population.
    """
    population = select_adj_control_population(
        train_info,
        use_stale_trust,
    )
    return population["graph_ratio"], population["factor_ratio"]


def should_stop_adj_ppo(
        epochs_ran,
        configured_epochs,
        graph_ratio,
        factor_ratio,
        graph_stop_ratio,
        factor_stop_ratio,
        min_epochs):
    """Return whether a trust violation actually truncates a later epoch."""
    epochs_ran = int(epochs_ran)
    configured_epochs = int(configured_epochs)
    min_epochs = int(min_epochs)
    if epochs_ran < min_epochs or epochs_ran >= configured_epochs:
        return False
    graph_stop = (
        float(graph_stop_ratio) > 0.0
        and math.isfinite(float(graph_ratio))
        and float(graph_ratio) >= float(graph_stop_ratio)
    )
    factor_stop = (
        float(factor_stop_ratio) > 0.0
        and math.isfinite(float(factor_ratio))
        and float(factor_ratio) >= float(factor_stop_ratio)
    )
    return graph_stop or factor_stop


def advance_recent_episode_window(
        current_window,
        configured_window,
        min_window,
        previous_graph_ratio,
        previous_factor_ratio,
        graph_stale_threshold,
        factor_stale_threshold,
        recover_graph_threshold,
        recover_factor_threshold,
        shrink_patience,
        recover_patience,
        severe_margin,
        emergency_window,
        emergency_graph_threshold,
        emergency_factor_threshold,
        high_count,
        low_count):
    """Advance the replay-window controller from one control population.

    Both ratios must describe the same population selected for the adjacency
    loss.  Non-finite ratios are accepted only for the initial controller
    state, before the first adjacency update has produced control metrics.
    """
    configured_window = max(1, int(configured_window))
    min_window = max(1, min(configured_window, int(min_window)))
    current_window = max(
        min_window,
        min(configured_window, int(current_window)),
    )
    original_window = current_window
    emergency_window = max(
        min_window,
        min(configured_window, int(emergency_window)),
    )
    shrink_patience = max(1, int(shrink_patience))
    recover_patience = max(1, int(recover_patience))
    high_count = max(0, int(high_count))
    low_count = max(0, int(low_count))
    graph_is_finite = math.isfinite(float(previous_graph_ratio))
    factor_is_finite = math.isfinite(float(previous_factor_ratio))
    if graph_is_finite != factor_is_finite:
        raise RuntimeError(
            "adjacency recent-window control ratios must become available "
            "together"
        )

    recovered = 0.0
    emergency_shrunk = 0.0
    if graph_is_finite:
        graph_ratio = float(previous_graph_ratio)
        factor_ratio = float(previous_factor_ratio)
        graph_too_stale = graph_ratio >= float(graph_stale_threshold)
        factor_too_stale = factor_ratio >= float(factor_stale_threshold)
        graph_fresh_enough = graph_ratio <= float(recover_graph_threshold)
        factor_fresh_enough = factor_ratio <= float(recover_factor_threshold)
        if graph_too_stale or factor_too_stale:
            high_count += 1
            low_count = 0
        elif graph_fresh_enough and factor_fresh_enough:
            low_count += 1
            high_count = 0
        else:
            high_count = 0
            low_count = 0

        if (
                graph_ratio >= float(emergency_graph_threshold)
                or factor_ratio >= float(emergency_factor_threshold)):
            current_window = emergency_window
            emergency_shrunk = float(current_window < original_window)
            high_count = 0
            low_count = 0
        elif high_count >= shrink_patience:
            if (
                    graph_ratio
                    >= float(graph_stale_threshold) + float(severe_margin)
                    or factor_ratio
                    >= float(factor_stale_threshold) + float(severe_margin)):
                current_window = min_window
            else:
                current_window = max(min_window, current_window - 1)
            high_count = 0
        elif (
                low_count >= recover_patience
                and current_window < configured_window):
            current_window = min(configured_window, current_window + 1)
            low_count = 0
            recovered = 1.0

    return {
        "window": current_window,
        "high_count": high_count,
        "low_count": low_count,
        # Event semantics: being below the configured maximum is a state, not
        # a new shrink.  Controllers and reports consume this field as a
        # per-update transition count, so repeated updates at the minimum must
        # remain zero.
        "shrunk": float(current_window < original_window),
        "recovered": recovered,
        "emergency_shrunk": emergency_shrunk,
    }


def validate_adj_control_application(
        use_stale_trust,
        raw_graph_ratio,
        raw_factor_ratio,
        trusted_graph_ratio,
        trusted_factor_ratio,
        control_graph_ratio,
        control_factor_ratio,
        epochs_ran,
        configured_epochs,
        early_stop_triggered,
        graph_stop_ratio,
        factor_stop_ratio,
        min_epochs):
    """Fail loudly when reporting and executed control semantics diverge."""
    if use_stale_trust:
        expected_graph = float(trusted_graph_ratio)
        expected_factor = float(trusted_factor_ratio)
        population_name = "trusted"
    else:
        expected_graph = float(raw_graph_ratio)
        expected_factor = float(raw_factor_ratio)
        population_name = "raw"
    values = (
        expected_graph,
        expected_factor,
        float(control_graph_ratio),
        float(control_factor_ratio),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(
            "adjacency {} control audit requires finite ratios".format(
                population_name
            )
        )
    if (
            abs(float(control_graph_ratio) - expected_graph) > 1e-12
            or abs(float(control_factor_ratio) - expected_factor) > 1e-12):
        raise RuntimeError(
            "adjacency {} control ratios do not match the executed "
            "population".format(population_name)
        )

    actual_truncation = int(epochs_ran) < int(configured_epochs)
    if bool(early_stop_triggered) != bool(actual_truncation):
        raise RuntimeError(
            "adjacency early-stop must report an actual truncation"
        )
    if actual_truncation and not should_stop_adj_ppo(
            epochs_ran=epochs_ran,
            configured_epochs=configured_epochs,
            graph_ratio=control_graph_ratio,
            factor_ratio=control_factor_ratio,
            graph_stop_ratio=graph_stop_ratio,
            factor_stop_ratio=factor_stop_ratio,
            min_epochs=min_epochs):
        raise RuntimeError(
            "adjacency update truncated without a matching control violation"
        )
    return True
