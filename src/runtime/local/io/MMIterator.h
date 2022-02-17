/*
 * Copyright 2022 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MM_FILE
#define MM_FILE

#include <iterator>
#include <cstddef>

#define MM_MAX_LINE_LENGTH 1025
#define MatrixMarketBanner "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH 64

#include <runtime/local/io/File.h>

typedef char MM_typecode[4];

char *mm_typecode_to_str(MM_typecode matcode);

/********************* MM_typecode query fucntions ***************************/

#define mm_is_matrix(typecode)	((typecode)[0]=='M')

#define mm_is_sparse(typecode)	((typecode)[1]=='C')
#define mm_is_coordinate(typecode)((typecode)[1]=='C')
#define mm_is_dense(typecode)	((typecode)[1]=='A')
#define mm_is_array(typecode)	((typecode)[1]=='A')

#define mm_is_complex(typecode)	((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode)	((typecode)[2]=='P')
#define mm_is_integer(typecode) ((typecode)[2]=='I')

#define mm_is_symmetric(typecode)((typecode)[3]=='S')
#define mm_is_general(typecode)	((typecode)[3]=='G')
#define mm_is_skew(typecode)	((typecode)[3]=='K')
#define mm_is_hermitian(typecode)((typecode)[3]=='H')

int mm_is_valid(MM_typecode matcode)
{
  if (!mm_is_matrix(matcode)) return 0;
  if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;
  if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;
  if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
    mm_is_skew(matcode))) return 0;
  return 1;
}


/********************* MM_typecode modify fucntions ***************************/

#define mm_set_matrix(typecode)	((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	((*typecode)[1]='C')
#define mm_set_array(typecode)	((*typecode)[1]='A')
#define mm_set_dense(typecode)	mm_set_array(typecode)
#define mm_set_sparse(typecode)	mm_set_coordinate(typecode)

#define mm_set_complex(typecode)((*typecode)[2]='C')
#define mm_set_real(typecode)	((*typecode)[2]='R')
#define mm_set_pattern(typecode)((*typecode)[2]='P')
#define mm_set_integer(typecode)((*typecode)[2]='I')


#define mm_set_symmetric(typecode)((*typecode)[3]='S')
#define mm_set_general(typecode)((*typecode)[3]='G')
#define mm_set_skew(typecode)	((*typecode)[3]='K')
#define mm_set_hermitian(typecode)((*typecode)[3]='H')

#define mm_clear_typecode(typecode) ((*typecode)[0]=(*typecode)[1]= \
									(*typecode)[2]=' ',(*typecode)[3]='G')

#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)


/********************* Matrix Market error codes ***************************/


#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17


/******************** Matrix Market internal definitions ********************

   MM_matrix_typecode: 4-character sequence

                      ojbect 	  sparse/    data       storage 
                                dense      type       scheme

   string position:	  [0]       [1]			  [2]         [3]

   Matrix typecode:   M(atrix)  C(oord)		R(eal)   	  G(eneral)
						                    A(array)	C(omplex)   H(ermitian)
											                    P(attern)   S(ymmetric)
								    		                  I(nteger)   K(kew)

 ***********************************************************************/


#define MM_MTX_STR		"matrix"
#define MM_ARRAY_STR	"array"
#define MM_DENSE_STR	"array"
#define MM_COORDINATE_STR "coordinate" 
#define MM_SPARSE_STR	"coordinate"
#define MM_COMPLEX_STR	"complex"
#define MM_REAL_STR		"real"
#define MM_INT_STR		"integer"
#define MM_GENERAL_STR  "general"
#define MM_SYMM_STR		"symmetric"
#define MM_HERM_STR		"hermitian"
#define MM_SKEW_STR		"skew-symmetric"
#define MM_PATTERN_STR  "pattern"

int mm_read_banner(File *f, MM_typecode *matcode){
  char *line;
  char banner[MM_MAX_TOKEN_LENGTH];
  char mtx[MM_MAX_TOKEN_LENGTH]; 
  char crd[MM_MAX_TOKEN_LENGTH];
  char data_type[MM_MAX_TOKEN_LENGTH];
  char storage_scheme[MM_MAX_TOKEN_LENGTH];
  char *p;


  mm_clear_typecode(matcode);  
  line = getLine(f);
  if (f->read == EOF) 
    return MM_PREMATURE_EOF;

  if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
      storage_scheme) != 5)
      return MM_PREMATURE_EOF;

  for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */
  for (p=crd; *p!='\0'; *p=tolower(*p),p++);  
  for (p=data_type; *p!='\0'; *p=tolower(*p),p++);
  for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);

  /* check for banner */
  if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
    return MM_NO_HEADER;

  /* first field should be "mtx" */
  if (strcmp(mtx, MM_MTX_STR) != 0)
    return  MM_UNSUPPORTED_TYPE;
  mm_set_matrix(matcode);


  /* second field describes whether this is a sparse matrix (in coordinate
          storage) or a dense array */


  if (strcmp(crd, MM_SPARSE_STR) == 0)
    mm_set_sparse(matcode);
  else
  if (strcmp(crd, MM_DENSE_STR) == 0)
    mm_set_dense(matcode);
  else
    return MM_UNSUPPORTED_TYPE;
    

  /* third field */

  if (strcmp(data_type, MM_REAL_STR) == 0)
    mm_set_real(matcode);
  else
  if (strcmp(data_type, MM_COMPLEX_STR) == 0)
    mm_set_complex(matcode);
  else
  if (strcmp(data_type, MM_PATTERN_STR) == 0)
    mm_set_pattern(matcode);
  else
  if (strcmp(data_type, MM_INT_STR) == 0)
    mm_set_integer(matcode);
  else
    return MM_UNSUPPORTED_TYPE;
    

  /* fourth field */

  if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
    mm_set_general(matcode);
  else
  if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
    mm_set_symmetric(matcode);
  else
  if (strcmp(storage_scheme, MM_HERM_STR) == 0)
    mm_set_hermitian(matcode);
  else
  if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
    mm_set_skew(matcode);
  else
    return MM_UNSUPPORTED_TYPE;
        

  return 0;
}

int mm_read_mtx_crd_size(File *f, size_t *M, size_t *N, size_t *nz )
{
  char* line;
  int num_items_read;

  /* set return null parameter values, in case we exit with errors */
  *M = *N = *nz = 0;
  do
  {
    num_items_read = sscanf(getLine(f), "%lu %lu %lu", M, N, nz);
    if (f->read == EOF) return MM_PREMATURE_EOF;
  } while (num_items_read != 3);

  return 0;
}

int mm_read_mtx_array_size(File *f, size_t *M, size_t *N)
{
    char *line;
    int num_items_read;
    /* set return null parameter values, in case we exit with errors */
    *M = *N = 0;
    do
    { 
      num_items_read = sscanf(getLine(f), "%lu %lu", M, N);
      if (f->read == EOF) return MM_PREMATURE_EOF;
    } while (num_items_read != 2);

    return 0;
}

template<typename VT>
class MMFile
{
private:
    File* f;
    MM_typecode typecode;
    size_t rows, cols, nnz;
public:
    MMFile(const char *filename){
        f = openFile(filename);
        mm_read_banner(f, &typecode);
        if (mm_is_coordinate(typecode))
            mm_read_mtx_crd_size(f, &rows, &cols, &nnz);
        else{
            mm_read_mtx_array_size(f, &rows, &cols);
            nnz = rows*cols;
        }
        std::cout << "Size is " << rows << "x" << cols << std::endl;
    }
    ~MMFile(){ /*closeFile(f);*/ }
    size_t numberRows() { return rows; }
    size_t numberCols() { return cols; }
    size_t entryCount() { return nnz; }

    struct Entry {
        size_t row, col;
        VT val;
        friend bool operator== (const Entry& a, const Entry& b) {
          return
          a.row == b.row &&
          a.col == b.col;
        }
        friend bool operator!=(const Entry& a, const Entry& b){
          return !(a==b);
        }
    };

    class MMIterator {
        using iterator_category = std::input_iterator_tag;
        using value_type  = Entry;
        using pointer     = Entry*;
        using reference   = Entry&;
        using difference_type = std::ptrdiff_t;
    private:
        pointer m_ptr;
        MMFile<VT> file;
        char *line;//[MM_MAX_LINE_LENGTH];
        size_t r = 0, c = 0;
        VT cur;
        MMIterator() {}
        void readEntry(){
          line = getLine(file.f);
          if(file.f->read == -1){
            terminate();
            return;
          }
          size_t pos = 0;
          if(mm_is_coordinate(file.typecode)){
              r = atoi(line);
              while(line[pos++] != ' ');
              c = atoi(line+pos);
              while(line[pos++] != ' ');
          }
          else if(r >= file.rows) { //Matrix is in array format
              r = 0; c++;
              //assert(c < cols && "Number of entries is greater than matrix size");
          }
          convertCstr(line + pos, &cur);
          *m_ptr = {r++, c, cur};
        }
    public:
        MMIterator(MMFile<VT>& f, bool read = true) : file(f) {
          if(read) readEntry();
        }
        void terminate() { *m_ptr = { -1ul, -1ul, cur}; }
        reference operator*() const { return *m_ptr; }
        pointer operator->() { return m_ptr; }

        // Prefix increment
        MMIterator& operator++() { readEntry(); return *this; }  

        // Postfix increment
        MMIterator operator++(int) { MMIterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const MMIterator& a, const MMIterator& b) { return *(a.m_ptr) == *(b.m_ptr); };
        friend bool operator!= (const MMIterator& a, const MMIterator& b) { return *(a.m_ptr) != *(b.m_ptr); };
    };

    MMIterator begin() {
      return MMIterator(*this);
    }
    MMIterator end() {
      MMIterator dummy = MMIterator(*this, false);
      dummy.terminate();
      return dummy;
    }
};

#endif //MM_FILE