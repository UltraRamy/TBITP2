import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class EliasGammaPostings:
    """
    Compression menggunakan Elias-Gamma Encoding, yang bersifat bit-level.

    Elias-Gamma encoding untuk integer positif n:
      1. Hitung k = floor(log2(n))
      2. Tulis k bit nol (unary prefix)
      3. Tulis satu bit '1' (separator)
      4. Tulis k bit terakhir dari n dalam biner (suffix, tanpa leading 1)

    Contoh:
      n=1: k=0 -> '1'
      n=2: k=1 -> '010'
      n=3: k=1 -> '011'
      n=6: k=2 -> '00110'

    Postings di-encode dengan gap-based (selisih antar doc ID yang berurutan).
    TF di-encode langsung (tidak gap-based).

    Karena bit-packing menghasilkan padding nol di akhir, jumlah integer
    disimpan sebagai 4-byte prefix agar decoding tahu kapan harus berhenti.
    """

    @staticmethod
    def _encode_number(n):
        """Encode satu integer positif n ke list of bits dengan Elias-Gamma."""
        if n <= 0:
            raise ValueError("Elias-Gamma hanya untuk integer positif (n >= 1)")
        k = n.bit_length() - 1
        suffix = n - (1 << k)
        bits = [0] * k + [1]
        for i in range(k - 1, -1, -1):
            bits.append((suffix >> i) & 1)
        return bits

    @staticmethod
    def _bits_to_bytes(bits):
        """Pack list of bits ke bytes, padding nol di akhir jika perlu."""
        remainder = len(bits) % 8
        if remainder:
            bits = bits + [0] * (8 - remainder)
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _bytes_to_bits(data):
        """Unpack bytes ke list of bits, MSB first."""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _decode_bits(bits, count):
        """Decode sejumlah 'count' integer dari list of bits."""
        numbers = []
        i = 0
        while len(numbers) < count and i < len(bits):
            k = 0
            while i < len(bits) and bits[i] == 0:
                k += 1
                i += 1
            if i >= len(bits):
                break
            i += 1
            suffix = 0
            for _ in range(k):
                if i < len(bits):
                    suffix = (suffix << 1) | bits[i]
                    i += 1
            numbers.append((1 << k) + suffix)
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list dengan gap-based Elias-Gamma.
        Format: [4-byte count][bit-packed Elias-Gamma encoded gaps]
        """
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i - 1])
        bits = []
        for n in gaps:
            bits.extend(EliasGammaPostings._encode_number(n))
        count_bytes = len(gaps).to_bytes(4, 'big')
        return count_bytes + EliasGammaPostings._bits_to_bytes(bits)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list dari bit-packed Elias-Gamma encoding.
        Kembalikan list of docIDs (bukan gaps).
        """
        count = int.from_bytes(encoded_postings_list[:4], 'big')
        bits = EliasGammaPostings._bytes_to_bits(encoded_postings_list[4:])
        gaps = EliasGammaPostings._decode_bits(bits, count)
        result = [gaps[0]]
        for i in range(1, len(gaps)):
            result.append(result[-1] + gaps[i])
        return result

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode TF list dengan Elias-Gamma (langsung, tidak gap-based).
        Format: [4-byte count][bit-packed Elias-Gamma encoded TFs]
        """
        bits = []
        for n in tf_list:
            bits.extend(EliasGammaPostings._encode_number(n))
        count_bytes = len(tf_list).to_bytes(4, 'big')
        return count_bytes + EliasGammaPostings._bits_to_bytes(bits)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """Decode TF list dari bit-packed Elias-Gamma encoding."""
        count = int.from_bytes(encoded_tf_list[:4], 'big')
        bits = EliasGammaPostings._bytes_to_bits(encoded_tf_list[4:])
        return EliasGammaPostings._decode_bits(bits, count)


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
