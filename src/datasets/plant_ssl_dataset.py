from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from m4.common import sha256


class BalancedPlantSSLDataset(Dataset):
    def __init__(self, sugar, external_rows, external_root, source_proportions):
        if not len(external_rows) or set(external_rows.split) != {"train"}:
            raise ValueError("Expanded SSL requires external training geometry")
        if external_rows.duplicated("geometry_sha256").any():
            raise ValueError("Duplicate external geometry is not independent data")
        self.sugar, self.rows = sugar, external_rows.reset_index(drop=True)
        self.clouds = []
        for row in self.rows.itertuples(index=False):
            cache_path = Path(external_root) / row.cache_file
            if hasattr(row, "cache_sha256") and sha256(cache_path) != row.cache_sha256:
                raise ValueError("External cache hash mismatch")
            with np.load(cache_path, allow_pickle=False) as archive:
                if not {"points", "center_indices", "neighbors"}.issubset(archive.files):
                    raise ValueError("Invalid XYZ cache")
                self.clouds.append({name: archive[name].copy() for name in ("points", "center_indices", "neighbors")})
        self.sources = ["Sugar4D"] + sorted(self.rows.source.unique())
        self.proportions = torch.tensor([source_proportions[name] for name in self.sources], dtype=torch.double)
        if not torch.isclose(self.proportions.sum(), torch.tensor(1., dtype=torch.double)):
            raise ValueError("Sampling source proportions must sum to one")
        self.groups = {"Sugar4D": {"sugar_beet": list(range(len(sugar)))}}
        self.identity_groups = {}
        for source in self.sources[1:]:
            frame = self.rows.loc[self.rows.source.eq(source)]
            self.groups[source] = {species: (group.index.to_numpy() + len(sugar)).tolist()
                                   for species, group in frame.groupby("species")}
            for species, group in frame.groupby("species"):
                if "identity_status" in group and "filename plant suffix" in str(group.identity_status.iloc[0]):
                    self.identity_groups[(source, species)] = [(plants.index.to_numpy() + len(sugar)).tolist()
                                                               for _, plants in group.groupby("plant_uid")]
        self.exposures = {source: {species: 0 for species in groups} for source, groups in self.groups.items()}

    def __len__(self):
        return len(self.sugar) + len(self.clouds)

    def __getitem__(self, index):
        if index < len(self.sugar):
            sample = self.sugar[index]
            return {name: sample[name] for name in ("points", "center_indices", "neighbors")}
        return {name: torch.from_numpy(values.copy()) for name, values in self.clouds[index - len(self.sugar)].items()}

    def sample_indices(self, count, generator):
        sampled_sources = torch.multinomial(self.proportions, count, replacement=True, generator=generator).tolist()
        result = []
        for source_index in sampled_sources:
            source = self.sources[source_index]
            species = sorted(self.groups[source])
            selected_species = species[int(torch.randint(len(species), (), generator=generator))]
            group = self.groups[source][selected_species]
            if (source, selected_species) in self.identity_groups:
                identities = self.identity_groups[(source, selected_species)]
                group = identities[int(torch.randint(len(identities), (), generator=generator))]
            result.append(group[int(torch.randint(len(group), (), generator=generator))])
            self.exposures[source][selected_species] += 1
        return result
