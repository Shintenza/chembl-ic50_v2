def build_extraction_query(offset: int, batch_size: int) -> str:
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
        AND act.potential_duplicate = 0
        AND a.confidence_score > 6
        AND td.chembl_id = 'CHEMBL203'
    ORDER BY
        act.activity_id
    LIMIT  {batch_size}
    OFFSET {offset};
    """
    return query
