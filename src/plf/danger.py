from .utils import Db
from .experiment import get_ppls, PipeLine

def corrupt_ppl(pplid: str):
    """
    Deletes a record from the 'ppls' table in the SQLite database if the provided
    `pplid` exists and the user confirms the deletion.

    Args:
        pplid (str): The ID of the person or record to be deleted.

    Raises:
        FileNotFoundError: If the database directory doesn't exist.
        sqlite3.Error: If there is an issue executing the SQL query.

    This function will:
    - Check if the provided `pplid` exists in the database.
    - Ask the user to confirm the deletion by entering the same `pplid`.
    - If the `pplid` matches, the record is deleted from the database.
    - If the `pplid` does not match, the deletion is aborted.
    - If the `pplid` is not found in the list, a message is displayed.
    """
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
