from .utils import Db
from .experiment import get_ppls, PipeLine

def corrupt_ppl(pplid: str):
    P = PipeLine()
    db_path = f"{P.settings['data_path']}/ppls.db"
    # Use the Db class context manager for automatic connection management
    with Db(db_path=db_path) as db:
        # Ensure the pplid exists in the database before attempting deletion
        if pplid in get_ppls():
            print('Cross verify before deleting.')

            # Single attempt to verify the correct pplid
            pplid1 = input("Enter the same pplid: ")
            if pplid == pplid1:
                try:
                    # Perform the deletion (no need for db.commit() since execute handles it)
                    db.execute("DELETE FROM ppls WHERE pplid = ?", (pplid,))
                    print(f"Record with pplid {pplid} has been corupted.")
                except Exception as e:
                    # In case there's an error, print the error message
                    print(f"Error deleting record: {e}")
            else:
                print("pplid does not match. Deletion aborted.")
        else:
            print(f"pplid {pplid} not found in the list of available pplids.")
