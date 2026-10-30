import React from "react";

export const MutationInspectorPage: React.FC = () => {
  return (
    <div className="workspace-container">
      <div className="page-header">
        <div className="page-title-row">
          <h2 className="page-title">Stage 2: Mutation Inspector</h2>
          <span className="badge badge-ready">Phase 2 Scheduled</span>
        </div>
        <p className="page-description">
          Validate mutation notation, verify amino acid residue identity against parent sequence,
          and align mutations to 5XJH crystal structure numbering.
        </p>
      </div>

      <div className="stub-card">
        <div className="stub-icon">🔍</div>
        <h3 className="stub-title">Mutation Inspector UI Scheduled for Phase 2</h3>
        <p className="stub-desc">
          The underlying scientific functions in <span className="path-badge">src/qcatalyst/data/mutation.py</span>{" "}
          are implemented and fully tested. Interactive sequence input and live 3D residue coordinate alignment
          will be exposed in the Phase 2 UI release.
        </p>

        <div className="stub-details">
          <h4>Available Scientific Implementations</h4>
          <ul>
            <li>
              <span className="func">parse_mutation_token(token)</span> — Single & multiple mutation token parser
            </li>
            <li>
              <span className="func">validate_mutations(seq, mutations)</span> — Residue bounds & parent identity checks
            </li>
            <li>
              <span className="func">map_mutation_positions_to_5xjh(mutations, enzyme)</span> — Crystal structure coordinate mapping
            </li>
            <li>
              <span className="func">build_and_save_variants_table()</span> — Curated table generator (430 variants ready)
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
