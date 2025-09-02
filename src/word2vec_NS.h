// same with ver1, but to avoid confusion
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_VOCAB_SIZE 30000000 // maximum vocabulary size 30M
#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

struct WORD { // idx is id(hash table)
    int count;
    char word[MAX_STRING];
    char* code;
    int codelen;
    int* point;
};

int getWordHash(char* word);
void buildHash(char* file_name);
void reduceHash();
void resetHashTable();
void initUnigramTable();

int readSentenceFromFile(FILE* fp, long long* sentence, long long id, int iter);
int searchVocabID(char* word);
char* IDtoWord(int id);


void initModel(); // not used
