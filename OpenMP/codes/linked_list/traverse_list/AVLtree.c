

#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// ······················································
// type definition
//

// define the node type
typedef struct nodet
{
  int key, depth;
  struct nodet *left;
  struct nodet *right;
} node_t;



// ······················································
//  functions prototypes
//

inline int      max          ( int, int );
inline int      get_depth    ( node_t * );
inline int      get_imbalance( node_t * );
inline node_t * make_node    ( int );

       void     insert             (node_t ** , int);
       node_t * insert_recursive   (node_t *, int);
       void     traverse_pre_order (node_t *);
       void     traverse_in_order  (node_t *);
       void     traverse_post_order(node_t *);


// ······················································
//  functions definitions
//

// »»»»»»»»»»»»»»»»»»
// helper routines

int max(int x, int y)
{ return (x > y)? x : y; }


int get_depth( node_t *node )
{ return ( node == NULL ? 0 : node->depth); }


int get_imbalance( node_t *node )
{ return ( node == NULL ? 0 : get_depth(node->left) - get_depth(node->right) ); }


node_t * make_node(int key)
{
  node_t *node = (node_t *)malloc(sizeof(node_t));
  node->key   = key;
  node->depth = 1;
  node->left   = NULL;
  node->right  = NULL;
  return(node);
}

// »»»»»»»»»»»»»»»»»»
//  rotation routines

node_t * Rrotate( node_t *node )
/*
 *      (node)        (nl)
         /            /  \
       (nl)    >>  (nlr) (node)
         \
	(nlr)
 */
{
    node_t *nl = node->left;
    node_t *nlr = nl->right;

    // Perform rotation
    nl->right = node;
    node->left = nlr;

    // Update heights
    node->depth = max(get_depth(node->left),
		      get_depth(node->right)) + 1;
    nl->depth   = max(get_depth(nl->left),
		      get_depth(nl->right)) + 1;

    // Return new root
    return nl;
}

node_t * Lrotate( node_t *node )
/*
       (node)           (nr)
          \             /  \  
	  (nr)  >>  (node) (nrl)
	  /
       (nrl)
 */
{
    node_t *nr  = node->right;
    node_t *nrl = nr->left;

    // Perform rotation
    nr->left = node;
    node->right = nrl;

    //  Update heights
    node->depth = max(get_depth(node->left),   
		       get_depth(node->right)) + 1;
    nr->depth   = max(get_depth(nr->left),
		      get_depth(nr->right)) + 1;

    // Return new root
    return nr;
}


// »»»»»»»»»»»»»»»»»»
//  insertion

void insert(node_t ** root, int key)
{
  typedef struct
  { node_t *ancestor, **branch; } stack_t;
      
  node_t *new = make_node(key);
  if (*root == NULL) {
    *root = new;
    return; }

  // As first, we find the insertion point
  // for a new leaf
  //
  
  int      depth = (*root)->depth;
  stack_t  stack[depth];
  node_t  *ptr = *root;
  int      idx = -1;
  
  while( ptr != NULL )
    {
      stack[++idx].ancestor = ptr;
      if ( key < ptr->key ) {
	stack[idx].branch = &(ptr->left);
	ptr = ptr->left; }
      else {
	stack[idx].branch = &(ptr->right);
	ptr = ptr->right; }
    }

  // Second, walk up the stack of
  // ancestors and check the imbalance
  //
  ptr = new;  
  while ( idx > -1 )
    {
      *(stack[idx].branch) = ptr;
      
      node_t *node = stack[idx].ancestor;
      node->depth = 1 + max(get_depth(node->left), get_depth(node->right));
      int imbalance = get_imbalance(node);

      if ( (imbalance > 1) && (imbalance < -1) )
	{
	
	  // Left Left Case
	  if ( imbalance > 1 && key < (node->left->key) )
	    ptr = Rrotate(node);
	  
	  // Right Right Case
	  if (imbalance < -1 && key > (node->right->key) )
	    ptr = Lrotate(node);
	  
	  // Left Right Case
	  if (imbalance > 1 && key > (node->left->key) )
	    {
	      node->left =  Lrotate(node->left);
	      ptr = Rrotate(node);
	    }

	  // Right Left Case
	  if (imbalance < -1 && key < (node->right->key) )
	    {
	      node->right = Rrotate(node->right);
	      ptr = Lrotate(node);
	    }
	}
      else
	ptr = node;
      
      idx--;
    }

  return;
}


node_t * insert_recursive(node_t * node, int key)
{

  // As first, we insert the node as a new leaf
  //
  if (node == NULL)
    // the bottom of the tree has been reached.
    // let's create a node and insert it here;
    // then we'll walk back
    return  make_node(key);
  
  if (key < node->key)  
    node->left  = insert_recursive(node->left, key);
  else if (key > node->key)
    node->right = insert_recursive(node->right, key);
  else
    return node;

  // Second, let's update the immediate ancestor
  // and find its new imbalance
    
  node->depth = 1 + max(get_depth(node->left), get_depth(node->right));
  int imbalance = get_imbalance(node);

  if ( (imbalance <= 1) && (imbalance >= -1) )
    return node;
  
  // Third, if the current node is imbalanced
  // we undergo appropriate adjustements
  
  // Left Left Case
  if ( imbalance > 1 && key < (node->left->key) )
    return Rrotate(node);

  // Right Right Case
  if (imbalance < -1 && key > (node->right->key) )
    return Lrotate(node);

  // Left Right Case
  if (imbalance > 1 && key > (node->left->key) )
    {
      node->left =  Lrotate(node->left);
      return Rrotate(node);
    }

  // Right Left Case
  if (imbalance < -1 && key < (node->right->key) )
    {
      node->right = Rrotate(node->right);
      return Lrotate(node);
    }

}


// »»»»»»»»»»»»»»»»»»
//  traversal

// Perform pre-order traversal
//
void traverse_pre_order(node_t *node)
{
  if(node != NULL)
    {
      printf("%d ", node->key);
      traverse_pre_order(node->left);
      traverse_pre_order(node->right);
    }
}


// Perform in-order traversal
//
void traverse_in_order(node_t *node)
{
  if(node != NULL)
    {      
      traverse_in_order(node->left);
      printf("%d ", node->key);
      traverse_in_order(node->right);
    }
}


// Perform in-order traversal
//
void traverse_post_order(node_t *node)
{
  if(node != NULL)
    {      
      traverse_post_order(node->left);
      traverse_post_order(node->right);
      printf("%d ", node->key);      
    }
}


/*
 *  ------------------------------------
 *  ------------------------------------
 */ 


int main( int argc, char **argv )

 #define RECURSIVE 0
 #define ITERATIVE 1
  
 #define PRE_ORDER 0
 #define IN_ORDER 1
 #define POST_ORDER 2
{
  int insertion = (argc > 1? atoi(*(argv+1)) : RECURSIVE );
  int order     = (argc > 2? atoi(*(argv+2)) : PRE_ORDER );
  node_t *root = NULL;
  //
  // build the tree

  if ( insertion == RECURSIVE )
    {
      root = insert_recursive( root, 10 );
      root = insert_recursive( root, 50 );
      root = insert_recursive( root, 20 );
      root = insert_recursive( root, 17 );
      root = insert_recursive( root, 87 );
      root = insert_recursive( root, 43 );
      root = insert_recursive( root, 6 );
      root = insert_recursive( root, 69 );
      root = insert_recursive( root, 55 );
    }
  else
    {
  
      insert( &root, 10 );
      insert( &root, 50 );
      insert( &root, 20 );
      insert( &root, 17 );
      insert( &root, 87 );
      insert( &root, 43 );
      insert( &root, 6 );
      insert( &root, 69 );
      insert( &root, 55 );
    }
  
  //
  
  
  if ( order == PRE_ORDER )
    traverse_pre_order(root);
  else if ( order == IN_ORDER )
    traverse_in_order(root);
  else if ( order == POST_ORDER )
      traverse_post_order(root);

  printf("\n");
  return 0;
}
