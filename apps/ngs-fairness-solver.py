import marimo

__generated_with = "0.14.7"
app = marimo.App()


@app.cell
def _():
    import pulp
    import pandas as pd
    import os
    return os, pd, pulp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load Input File""")
    return


@app.cell
def _(os, pd):
    # Files base dir
    FILE_BASE_DIR = 'apps/ngs-fairness-solver-files'
    # Input file path
    excel_file = os.path.join(FILE_BASE_DIR,'ngs_fairness_solver_input_tmpl.xlsx')  # Change this to your actual file name

    # Load period demands
    periods_df = pd.read_excel(excel_file, sheet_name='Periods')
    # columns: ['period', 'SA', 'A']

    # Load target proportions
    targets_df = pd.read_excel(excel_file, sheet_name='Target Proportions')
    # columns: ['entity_type', 'entity_code', 'target_proportions']

    # Extract sets
    periods = periods_df['period'].tolist()
    ranks = ['SA', 'A']
    watches = targets_df[targets_df['entity_type'] == 'watch']['entity_code'].tolist()
    divisions = targets_df[targets_df['entity_type'] == 'division']['entity_code'].tolist()

    # Target proportions
    watch_targets = targets_df[targets_df['entity_type'] == 'watch'].set_index('entity_code')['target_proportion'].to_dict()
    division_targets = targets_df[targets_df['entity_type'] == 'division'].set_index('entity_code')['target_proportion'].to_dict()
    return (
        FILE_BASE_DIR,
        division_targets,
        divisions,
        periods,
        periods_df,
        ranks,
        watch_targets,
        watches,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Set Tolerance Level""")
    return


@app.cell
def _():
    # Tolerance for overall ratio constraints (e.g., 2%)
    TOLERANCE = 0.00
    return (TOLERANCE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Build Model""")
    return


@app.cell
def _(
    TOLERANCE,
    division_targets,
    divisions,
    periods,
    periods_df,
    pulp,
    ranks,
    watch_targets,
    watches,
):
    model = pulp.LpProblem("Fair_Sparse_Deployment", pulp.LpMinimize)

    # Decision variables: x[rank, watch, division, period]
    x = pulp.LpVariable.dicts(
        "x",
        ((rank, w, d, p) for rank in ranks for w in watches for d in divisions for p in periods),
        lowBound=0,
        cat='Integer'
    )

    # Max concentration per period and rank (for sparsity)
    z = pulp.LpVariable.dicts(
        "z",
        ((rank, p) for rank in ranks for p in periods),
        lowBound=0,
        cat='Integer'
    )

    # Slack variables for intra-division watch ratios (across all periods)
    slack_divwatch_pos = pulp.LpVariable.dicts(
        "slack_divwatch_pos",
        ((rank, d, w) for rank in ranks for d in divisions for w in watches),
        lowBound=0,
        cat='Continuous'
    )

    slack_divwatch_neg = pulp.LpVariable.dicts(
        "slack_divwatch_neg",
        ((rank, d, w) for rank in ranks for d in divisions for w in watches),
        lowBound=0,
        cat='Continuous'
    )

    for idx, p in enumerate(periods):
        for rank in ranks:
            period_demand = periods_df.loc[idx, rank]
            model += (pulp.lpSum((x[rank, w, d, p] for w in watches for d in divisions)) == period_demand, f'Demand_{rank}_{p}')

    for rank in ranks:
        total_rank = periods_df[rank].sum()
        for w in watches:
            target = watch_targets[w] * total_rank
            tol = max(1, int(round(TOLERANCE * total_rank)))
            actual = pulp.lpSum((x[rank, w, d, p] for d in divisions for p in periods))
            model += (actual >= target - tol, f'WatchLow_{rank}_{w}')
            model += (actual <= target + tol, f'WatchHigh_{rank}_{w}')

    for rank in ranks:
        total_rank = periods_df[rank].sum()
        for d in divisions:
            target = division_targets[d] * total_rank
            tol = max(1, int(round(TOLERANCE * total_rank)))
            actual = pulp.lpSum((x[rank, w, d, p] for w in watches for p in periods))
            model += (actual >= target - tol, f'DivLow_{rank}_{d}')
            model += (actual <= target + tol, f'DivHigh_{rank}_{d}')

    for rank in ranks:
        for p in periods:
            for w in watches:
                for d in divisions:
                    model += (x[rank, w, d, p] <= z[rank, p], f'Sparse_{rank}_{w}_{d}_{p}')

    for rank in ranks:
        total_rank = periods_df[rank].sum()
        for d in divisions:
            total_div = division_targets[d] * total_rank
            for w in watches:
                target = watch_targets[w] * total_div
                actual = pulp.lpSum((x[rank, w, d, p] for p in periods))
                model += (actual - target <= slack_divwatch_pos[rank, d, w])
                model += (target - actual <= slack_divwatch_neg[rank, d, w])

    model += (pulp.lpSum((z[rank, p] for rank in ranks for p in periods)) + 0.1 * pulp.lpSum((slack_divwatch_pos[rank, d, w] + slack_divwatch_neg[rank, d, w] for rank in ranks for d in divisions for w in watches)), 'Total_Objective')
    return model, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Solve""")
    return


@app.cell
def _(model, pulp):
    solver = pulp.PULP_CBC_CMD(msg=True)
    result_status = model.solve(solver)
    print('Solver status:', pulp.LpStatus[model.status])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Display Results""")
    return


@app.cell
def _():
    ordered_divisions = ['HOS', 'KWL', 'KCE', 'NTE', 'NTW']
    ordered_watches = ['A', 'B', 'C', 'D', 'E']
    return ordered_divisions, ordered_watches


@app.cell
def _(
    division_targets,
    divisions,
    model,
    ordered_divisions,
    ordered_watches,
    pd,
    periods,
    pulp,
    ranks,
    watch_targets,
    watches,
    x,
):
    if pulp.LpStatus[model.status] == 'Optimal':
        deployment_data = []
        for rank_1 in ranks:
            for p_1 in periods:
                for w_1 in watches:
                    for d_1 in divisions:
                        val = int(pulp.value(x[rank_1, w_1, d_1, p_1]))
                        deployment_data.append({'period': p_1, 'rank': rank_1, 'watch': w_1, 'division': d_1, 'count': val})
        results_df = pd.DataFrame(deployment_data)
        for rank_1 in ranks:
            print(f"\n{'=' * 30}\n{rank_1} DEPLOYMENTS\n{'=' * 30}")
            rank_df = results_df[results_df['rank'] == rank_1]
            for p_1 in periods:
                print(f'\nPeriod {p_1}:')
                period_df = rank_df[rank_df['period'] == p_1]
                grid = pd.DataFrame(0, index=ordered_divisions, columns=ordered_watches)
                for _, row in period_df.iterrows():
                    grid.at[row['division'], row['watch']] = row['count']
                grid.loc['TOTAL'] = grid.sum()
                grid['TOTAL'] = grid.sum(axis=1)
                print(grid)
        for rank_1 in ranks:
            print(f"\n{'=' * 30}\n{rank_1} OVERALL PROPORTIONS\n{'=' * 30}")
            rank_df = results_df[results_df['rank'] == rank_1]
            total = rank_df['count'].sum()
            print('Watch proportions:')
            watch_actual = rank_df.groupby('watch')['count'].sum() / total
            for w_1 in watches:
                print(f'  {w_1}: Actual={watch_actual.get(w_1, 0):.2%}  Target={watch_targets[w_1]:.2%}  Delta={watch_actual.get(w_1, 0) - watch_targets[w_1]:+.2%}')
            print('Division proportions:')
            div_actual = rank_df.groupby('division')['count'].sum() / total
            for d_1 in divisions:
                print(f'  {d_1}: Actual={div_actual.get(d_1, 0):.2%}  Target={division_targets[d_1]:.2%}  Delta={div_actual.get(d_1, 0) - division_targets[d_1]:+.2%}')
        print('\n' + '=' * 50)
        print('INTRA-DIVISION WATCH RATIO SUMMARY')
        print('=' * 50)
        for rank_1 in ranks:
            print(f"\n{'=' * 30}\n{rank_1} INTRA-DIVISION WATCH RATIOS\n{'=' * 30}")
            rank_df = results_df[results_df['rank'] == rank_1]
            for d_1 in divisions:
                div_df = rank_df[rank_df['division'] == d_1]
                total_in_div = div_df['count'].sum()
                print(f'\nDivision {d_1} (Total assigned: {total_in_div}):')
                summary = []
                for w_1 in watches:
                    actual_1 = div_df[div_df['watch'] == w_1]['count'].sum()
                    actual_prop = actual_1 / total_in_div if total_in_div > 0 else 0
                    target_prop = watch_targets[w_1]
                    delta = actual_prop - target_prop
                    summary.append({'Watch': w_1, 'Actual': f'{actual_prop:.2%}', 'Target': f'{target_prop:.2%}', 'Delta': f'{delta:+.2%}'})
                summary_df = pd.DataFrame(summary).set_index('Watch')
                print(summary_df)
    else:
        print('No optimal solution found. Try increasing TOLERANCE.')
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Save Result to xlsx""")
    return


@app.cell
def _(
    FILE_BASE_DIR,
    division_targets,
    ordered_divisions,
    ordered_watches,
    os,
    pd,
    periods,
    ranks,
    results_df,
    watch_targets,
):
    output_excel = os.path.join(FILE_BASE_DIR,'deployment_report.xlsx')
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for p_2 in periods:
            for rank_2 in ranks:
                rank_df_1 = results_df[results_df['rank'] == rank_2]
                period_df_1 = rank_df_1[rank_df_1['period'] == p_2]
                grid_1 = pd.DataFrame(0, index=ordered_divisions, columns=ordered_watches)
                for _, row_1 in period_df_1.iterrows():
                    grid_1.at[row_1['division'], row_1['watch']] = row_1['count']
                grid_1.loc['TOTAL'] = grid_1.sum()
                grid_1['TOTAL'] = grid_1.sum(axis=1)
                sheet_name = f'{rank_2}_{p_2}'
                grid_1.to_excel(writer, sheet_name=sheet_name)
            total_1 = rank_df_1['count'].sum()
            watch_actual_1 = rank_df_1.groupby('watch')['count'].sum() / total_1
            watch_table = pd.DataFrame({'Actual': watch_actual_1, 'Target': pd.Series(watch_targets), 'Delta': watch_actual_1 - pd.Series(watch_targets)})
            watch_table.index.name = 'Watch'
            watch_table = watch_table.apply(lambda col: col.map(lambda x: f'{x:.2%}' if isinstance(x, float) else x))
            watch_table.to_excel(writer, sheet_name=f'{rank_2}_Watch_Proportions')
            div_actual_1 = rank_df_1.groupby('division')['count'].sum() / total_1
            div_table = pd.DataFrame({'Actual': div_actual_1, 'Target': pd.Series(division_targets), 'Delta': div_actual_1 - pd.Series(division_targets)})
            div_table.index.name = 'Division'
            div_table = div_table.apply(lambda col: col.map(lambda x: f'{x:.2%}' if isinstance(x, float) else x))
            div_table.to_excel(writer, sheet_name=f'{rank_2}_Division_Proportions')
    print(f'\nAll deployment matrices and proportion tables exported to {output_excel}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
