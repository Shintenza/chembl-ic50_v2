"""
SQL query builder for ChEMBL IC50 extraction.

Filters are applied at the database level to minimise network transfer.
The query returns one row per activity record; duplicate activities are
excluded via the ``potential_duplicate = 0`` predicate so no further
deduplication step is required downstream.
"""


def build_extraction_query(offset: int, batch_size: int) -> str:
    """Return a SQL SELECT statement for a single paginated batch.

    Parameters
    ----------
    offset:
        Number of rows to skip (OFFSET clause).
    batch_size:
        Maximum number of rows to return (LIMIT clause).

    Returns
    -------
    str
        A complete, parameterised-free SQL string ready to execute via
        ``cursor.execute(query)``.  All filter values are embedded as
        literals because they are hard configuration constants, not
        user-supplied input.
    """
    query = f"""
SELECT
    act.activity_id,
    act.molregno,
    md.chembl_id                  AS compound_chembl_id,
    cs.canonical_smiles,
    act.standard_type,
    act.standard_relation,
    act.standard_value,
    act.standard_units,
    act.pchembl_value,
    act.data_validity_comment,
    act.assay_id,
    a.assay_type,
    a.confidence_score,
    a.chembl_id                   AS assay_chembl_id,
    td.chembl_id                  AS target_chembl_id,
    td.pref_name                  AS target_name,
    td.target_type,
    td.organism
FROM
    activities            act
    JOIN assays           a   ON act.assay_id      = a.assay_id
    JOIN target_dictionary td  ON a.tid             = td.tid
    JOIN molecule_dictionary md ON act.molregno     = md.molregno
    JOIN compound_structures cs ON act.molregno     = cs.molregno
WHERE
    act.standard_type     = 'IC50'
    AND act.standard_units    = 'nM'
    AND act.standard_relation = '='
    AND act.pchembl_value     IS NOT NULL
    AND act.standard_value    > 0
    AND (
        act.data_validity_comment IS NULL
        OR act.data_validity_comment = 'Manually validated'
    )
    AND act.potential_duplicate = 0
    AND a.assay_type       IN ('B', 'F')
    AND a.confidence_score  = 9
    AND td.target_type      = 'SINGLE PROTEIN'
ORDER BY
    act.activity_id
LIMIT  {batch_size}
OFFSET {offset}
;
"""
    return query
