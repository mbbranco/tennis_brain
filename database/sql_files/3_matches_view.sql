DROP VIEW IF EXISTS matches_view;

CREATE VIEW matches_view AS
SELECT
    id,
    tourney_id,
    tourney_name,
    surface,
    draw_size,
    tourney_level,
    tourney_date as tourney_date_int,
    match_num,
    winner_id,
    winner_seed,
    winner_entry,
    winner_name,
    winner_hand,
    winner_ht,
    winner_ioc,
    winner_age,
    loser_id,
    loser_seed,
    loser_entry,
    loser_name,
    loser_hand,
    loser_ht,
    loser_ioc,
    loser_age,
    score,
    best_of,
    round,
    minutes,
    w_ace,
    w_df,
    w_svpt,
    w_1stIn,
    w_1stWon,
    w_2ndWon,
    w_SvGms,
    w_bpSaved,
    w_bpFaced,
    l_ace,
    l_df,
    l_svpt,
    l_1stIn,
    l_1stWon,
    l_2ndWon,
    l_SvGms,
    l_bpSaved,
    l_bpFaced,
    winner_rank,
    winner_rank_points,
    loser_rank,
    loser_rank_points,
    SUBSTR(tourney_date, 1, 4) || '-' || substr(tourney_date, 5, 2) || '-' || substr(tourney_date, 7, 2) as tourney_date,
    CAST(SUBSTR(tourney_date, 1, 4) AS INTEGER) as tourney_year,
    CASE
        WHEN tourney_level='A' THEN 250
        WHEN tourney_level = 'M' THEN 1000
        WHEN tourney_level = 'G' THEN 2000 
        ELSE 0
    END as tourney_points,
    CASE
        WHEN round ='RR' THEN 7
        WHEN round = 'R128' THEN 6
        WHEN round = 'R64' THEN 5
        WHEN round = 'R32' THEN 4
        WHEN round = 'R16' THEN 3
        WHEN round = 'QF' THEN 2
        WHEN round = 'SF' THEN 1
        WHEN round = 'F' THEN 0
        ELSE 7
    END as round_level
FROM matches
WHERE 
    tourney_date >= 20100101
    AND tourney_level NOT IN ('C','S','F','D','P','PM','I','E','J','T')
    AND tourney_name NOT LIKE '%Olympics%'
    AND tourney_name NOT LIKE '%Cup%'
    AND tourney_name NOT LIKE '%Finals%'
ORDER BY 
    tourney_date ASC,
    match_num ASC;
    
SELECT 
    *
FROM matches_view;