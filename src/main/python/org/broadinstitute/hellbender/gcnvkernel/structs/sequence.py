from typing import Optional, List, Union


class SAMSequenceEntry:
    """Represents a SAM sequence dictionary header line."""
    def __init__(self,
                 name: str,
                 length: int,
                 url: Optional[str] = None,
                 assembly: Optional[str] = None,
                 md5: Optional[str] = None,
                 species: Optional[str] = None):
        assert name is not None, "Sequence name is necessary"
        assert length is not None, "Sequence length is necessary"
        self.name = name
        self.length = length
        self.url = url
        self.assembly = assembly
        self.md5 = md5
        self.species = species

    @staticmethod
    def _assert_not_none(entry: object, short_name: str, long_name: str):
        assert entry is not None, "SAM header entry {0} (\"{1}\") is not known".format(short_name, long_name)

    @property
    def SN(self):
        SAMSequenceEntry._assert_not_none(self.name, "SN", "name")
        return self.name

    @property
    def LN(self):
        SAMSequenceEntry._assert_not_none(self.length, "LN", "length")
        return self.length

    @property
    def UR(self):
        SAMSequenceEntry._assert_not_none(self.url, "UR", "URL")
        return self.url

    @property
    def AS(self):
        SAMSequenceEntry._assert_not_none(self.assembly, "AS", "assembly")
        return self.assembly

    @property
    def M5(self):
        SAMSequenceEntry._assert_not_none(self.md5, "M5", "MD5")
        return self.md5

    @property
    def SP(self):
        SAMSequenceEntry._assert_not_none(self.species, "SP", "species")
        return self.species

    def __repr__(self):
        return "SN:{0}\tLN:{1}\tUR:{2}\tAS:{3}\tM5:{4}\tSP:{5}".format(
            self.SN,
            repr(self.LN),
            self.url if self.url is not None else "n/a",
            self.assembly if self.assembly is not None else "n/a",
            self.md5 if self.md5 is not None else "n/a",
            self.species if self.species is not None else "n/a")


class SAMSequenceDictionary:
    def __init__(self, sam_header_entry_list: Optional[List[SAMSequenceEntry]] = None):
        if sam_header_entry_list is None:
            self._sam_header_entry_list: List[SAMSequenceEntry] = []
            self._seq_names: List[str] = []
        else:
            self._sam_header_entry_list = sam_header_entry_list.copy()
            self._seq_names = [entry.SN for entry in sam_header_entry_list]

    def append(self, sam_header_entry: SAMSequenceEntry):
        self._sam_header_entry_list.append(sam_header_entry)
        self._seq_names.append(sam_header_entry.SN)

    def __len__(self):
        return len(self._sam_header_entry_list)

    def __getitem__(self, index):
        return self._sam_header_entry_list[index]

    def index(self, name: Union[str, SAMSequenceEntry]):
        if isinstance(name, str):
            return self._seq_names.index(name)
        elif isinstance(name, SAMSequenceEntry):
            return self._sam_header_entry_list.index(name)
        else:
            raise Exception("Input object is not a valid type.")

    @staticmethod
    def from_file(input_file: str, max_lines: Optional[int] = None):
        seq_dict = SAMSequenceDictionary()

        def parse_header_element(_sam_header_values: List[str]) -> bool:
            name = None
            length = None
            url = None
            assembly = None
            md5 = None
            species = None

            for entry in _sam_header_values:
                separator_pos = entry.find(':')

                if separator_pos > 0:
                    key = entry[:separator_pos]
                    value = entry[separator_pos + 1:]
                else:
                    return False

                if key == 'SN':
                    name = value
                elif key == 'LN':
                    length = int(value)
                elif key == 'UR':
                    url = value
                elif key == 'AS':
                    assembly = value
                elif key == 'M5':
                    md5 = value
                elif key == 'SP':
                    species = value
                else:
                    return False

            seq_dict.append(SAMSequenceEntry(name, length, url, assembly, md5, species))
            return True

        with open(input_file, 'r') as f:
            for line_no, line in enumerate(f):
                stripped_line = line.strip()
                if len(stripped_line) == 0:
                    continue
                elif stripped_line[0] == '@' and len(stripped_line) > 3:
                    tokens = stripped_line[1:].split('\t')
                    sam_header_key = tokens[0]
                    sam_header_values = tokens[1:]
                    if sam_header_key == 'HD':
                        continue
                    elif sam_header_key == 'SQ':
                        if not parse_header_element(sam_header_values):
                            raise Exception("Error parsing SAM header line \"{0}\" in {1}".format(
                                stripped_line, input_file))
                    else:
                        raise Exception("Unknown SAM header key {0} in {1}.".format(sam_header_key, input_file))
                else:
                    if max_lines is not None and line_no < max_lines:
                        continue
                    else:
                        break

        if len(seq_dict) == 0:
            raise Exception("No SAM header could be found {0}.".format(input_file))
        else:
            return seq_dict

    def __repr__(self):
        if len(self._sam_header_entry_list) == 0:
            return "empty"
        else:
            out = ""
            for entry in self._sam_header_entry_list:
                out += repr(entry) + '\n'
            return out

