DROP VIEW IF EXISTS rankings_view;

CREATE VIEW rankings_view AS
    WITH all_rankings AS (
        SELECT
            id,
            ranking_date as 'ranking_date_int',
            rank,
            player,
            points,
            SUBSTR(ranking_date, 1, 4) || '-' || substr(ranking_date, 5, 2) || '-' || substr(ranking_date, 7, 2) as 'ranking_date',
            SUBSTR(ranking_date, 1, 4) || '-' || substr(ranking_date, 5, 2) as 'ranking_ym'
    
        FROM RANKINGS
    ),

    ranking_ym_date AS (
        SELECT
            MAX(ranking_date) as ranking_date
        FROM all_rankings
        GROUP BY ranking_ym
    )

    SELECT
        all_rankings.id,
        ranking_ym_date.ranking_date,
        all_rankings.rank,
        all_rankings.player,
        all_rankings.points
    FROM ranking_ym_date
    INNER JOIN all_rankings ON all_rankings.ranking_date = ranking_ym_date.ranking_date
    ORDER BY all_rankings.player, all_rankings.ranking_ym;

SELECT *  FROM rankings_view;
